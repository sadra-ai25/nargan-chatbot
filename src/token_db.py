"""
Token Management Database using SQLite
"""
import sqlite3
from pathlib import Path
from typing import Optional

# Database path
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tokens.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Default token allocation
DEFAULT_TOKENS = 10000


def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT UNIQUE NOT NULL,
            client_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    # Create tokens table - hardcode DEFAULT value
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            total_tokens INTEGER DEFAULT {DEFAULT_TOKENS},
            used_tokens INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Create token history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tokens_used INTEGER NOT NULL,
            question_preview TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()


def get_client_ip(request) -> str:
    """Extract client IP from request"""
    # Check for forwarded headers (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client host
    if hasattr(request, 'client') and request.client:
        return request.client.host
    
    return "unknown"


def get_or_create_user(ip_address: str, client_id: str = None) -> dict:
    """Get existing user or create new one"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT * FROM users WHERE ip_address = ?", (ip_address,))
    user = cursor.fetchone()
    
    if user:
        # Update last seen
        cursor.execute(
            "UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE ip_address = ?",
            (ip_address,)
        )
        conn.commit()
        
        # Get token info
        cursor.execute("SELECT * FROM tokens WHERE user_id = ?", (user['id'],))
        token = cursor.fetchone()
        
        conn.close()
        return {
            'user': dict(user),
            'tokens': dict(token) if token else None
        }
    
    # Create new user
    cursor.execute(
        "INSERT INTO users (ip_address, client_id) VALUES (?, ?)",
        (ip_address, client_id)
    )
    user_id = cursor.lastrowid
    
    # Allocate tokens
    cursor.execute(
        "INSERT INTO tokens (user_id, total_tokens) VALUES (?, ?)",
        (user_id, DEFAULT_TOKENS)
    )
    
    conn.commit()
    conn.close()
    
    return {
        'user': {'id': user_id, 'ip_address': ip_address},
        'tokens': {'total_tokens': DEFAULT_TOKENS, 'used_tokens': 0}
    }


def get_user_tokens(user_id: int) -> dict:
    """Get token balance for user"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tokens WHERE user_id = ?", (user_id,))
    token = cursor.fetchone()
    
    conn.close()
    
    if token:
        return dict(token)
    return {'total_tokens': DEFAULT_TOKENS, 'used_tokens': 0}


def calculate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    # Simple estimation: ~4 chars per token for Persian/English mix
    return max(1, len(text) // 4)


def use_tokens(user_id: int, question: str, answer: str) -> dict:
    """Deduct tokens for a query"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Calculate tokens used
    question_tokens = calculate_tokens(question)
    answer_tokens = calculate_tokens(answer)
    total_used = question_tokens + answer_tokens
    
    # Get current balance
    cursor.execute("SELECT * FROM tokens WHERE user_id = ?", (user_id,))
    token = cursor.fetchone()
    
    if not token:
        conn.close()
        return {'error': 'User not found', 'remaining': 0}
    
    current_used = token['used_tokens']
    total_tokens = token['total_tokens']
    remaining = total_tokens - current_used
    
    # Check if enough tokens
    if remaining < total_used:
        conn.close()
        return {
            'error': 'Insufficient tokens',
            'remaining': remaining,
            'required': total_used
        }
    
    # Update tokens
    new_used = current_used + total_used
    cursor.execute(
        "UPDATE tokens SET used_tokens = ? WHERE user_id = ?",
        (new_used, user_id)
    )
    
    # Log history
    cursor.execute(
        "INSERT INTO token_history (user_id, tokens_used, question_preview) VALUES (?, ?, ?)",
        (user_id, total_used, question[:100])
    )
    
    conn.commit()
    conn.close()
    
    return {
        'tokens_used': total_used,
        'remaining': total_tokens - new_used,
        'total_allocated': total_tokens
    }


def reset_user_tokens(user_id: int, new_amount: int = DEFAULT_TOKENS) -> dict:
    """Reset user tokens to default or specified amount"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE tokens SET total_tokens = ?, used_tokens = 0 WHERE user_id = ?",
        (new_amount, user_id)
    )
    
    conn.commit()
    conn.close()
    
    return {'total_tokens': new_amount, 'used_tokens': 0}


def get_user_history(user_id: int, limit: int = 50) -> list:
    """Get token usage history for user"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT * FROM token_history 
           WHERE user_id = ? 
           ORDER BY timestamp DESC 
           LIMIT ?""",
        (user_id, limit)
    )
    
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return history


def get_all_users() -> list:
    """Get all users with their token status"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT u.*, t.total_tokens, t.used_tokens,
               (t.total_tokens - t.used_tokens) as remaining
        FROM users u
        JOIN tokens t ON u.id = t.user_id
        ORDER BY u.last_seen DESC
    """)
    
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return users

def get_or_create_user_by_chat(chat_id: str, module: str = "documents"):
    """Get or create user by chat_id (for new system)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Try to find by client_id (chat_id)
    cursor.execute("SELECT * FROM users WHERE client_id = ?", (chat_id,))
    user = cursor.fetchone()
    
    if user:
        # Update last seen
        cursor.execute(
            "UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?",
            (user['id'],)
        )
        conn.commit()
        
        cursor.execute("SELECT * FROM tokens WHERE user_id = ?", (user['id'],))
        token = cursor.fetchone()
        conn.close()
        return {
            'user': dict(user),
            'tokens': dict(token) if token else None
        }
    
    # Create new user with chat_id as client_id
    cursor.execute(
        "INSERT INTO users (ip_address, client_id) VALUES (?, ?)",
        (chat_id, chat_id)
    )
    user_id = cursor.lastrowid
    
    cursor.execute(
        "INSERT INTO tokens (user_id, total_tokens) VALUES (?, ?)",
        (user_id, DEFAULT_TOKENS)
    )
    
    conn.commit()
    conn.close()
    
    return {
        'user': {'id': user_id, 'ip_address': chat_id, 'client_id': chat_id},
        'tokens': {'total_tokens': DEFAULT_TOKENS, 'used_tokens': 0}
    }


# Initialize database on import
init_database()