import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import bcrypt

from dotenv import load_dotenv

load_dotenv()


class Database:
    def __init__(self):
        self.connection = None
        # Database connection establish karne wala function 
        self.connect()

    # CONNECTION & RECONNECT
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "phishing_detector"),
                user=os.getenv("DB_USER", "root"),
                password=os.getenv("DB_PASSWORD", ""),
                autocommit=False,
                connection_timeout=30,
            )
            if self.connection.is_connected():
                print("Database connected")
                return True
        except Error as e:
            print(f"Connection error: {e}")
            self.connection = None
            return False
    
    # reconnect
    def ensure_connection(self):
        try:
            if self.connection is None or not self.connection.is_connected():
                print("Database connection is lost. Reconnecting...")
                self.connect()
            else:
                self.connection.ping(reconnect=True, attempts=3, delay=2)
        except Error:
            self.connect()
    
    # connection closing
    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")

    # Passwordd
    def hash_password(self, password: str) -> str:
        # generating salt
        salt = bcrypt.gensalt()
        # hashpw.. pswd+salt ko hash krta 
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    # user managment 
    def create_user(self, username: str, email: str, password: str):
        self.ensure_connection()
        # Database query execute karne ke liye
        cursor = self.connection.cursor()
        try:
            hashed = self.hash_password(password)
            query = """
            INSERT INTO users (user_name, email, password, created_at, is_active, total_scans)
                VALUES (%s, %s, %s, NOW(), TRUE, 0)
            """
            cursor.execute(query, (username, email, hashed))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            self.connection.rollback()
            print(f"create_user error: {e}")
            return None
        finally:
            cursor.close()

    def get_user_by_email(self, email: str):
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cursor.fetchone()
        except Error as e:
            print(f"get_user_by_email error: {e}")
            return None
        finally:
            cursor.close()

    def get_user_by_id(self, user_id: int):
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT user_id, user_name, email, created_at, last_login, total_scans FROM users WHERE user_id = %s",
                (user_id,),
            )
            return cursor.fetchone()
        except Error as e:
            print(f"get_user_by_id error: {e}")
            return None
        finally:
            cursor.close()

    def update_last_login(self, user_id: int):
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "UPDATE users SET last_login = NOW() WHERE user_id = %s", (user_id,)
            )
            self.connection.commit()
        except Error as e:
            self.connection.rollback()
            print(f"update_last_login error: {e}")
        finally:
            cursor.close()

    def update_user_total_scans(self, user_id: int):
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "UPDATE users SET total_scans = total_scans + 1 WHERE user_id = %s",
                (user_id,),
            )
            self.connection.commit()
        except Error as e:
            self.connection.rollback()
            print(f"update_user_total_scans error: {e}")
        finally:
            cursor.close()

    # SCAN MANAGEMENT
    def save_scan(
        self,
        user_id,
        original_text,
        processed_text,
        prediction,
        confidence,
        risk_level,
        processing_time,
        model_id=1,
    ):
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            query = """
                INSERT INTO email_scan
                    (user_id, model_id, original_text, processed_text,
                     prediction_result, confidence_score, risk_level,
                     scan_timestamp, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
            """
            values = (
                user_id,
                model_id,
                original_text[:5000],
                processed_text[:5000],
                prediction,
                confidence,
                risk_level,
                processing_time,
            )
            cursor.execute(query, values)
            self.connection.commit()
            scan_id = cursor.lastrowid
            if scan_id and user_id:
                self.update_user_total_scans(user_id)
            print(f"Scan saved — ID: {scan_id}")
            return scan_id
        except Error as e:
            self.connection.rollback()
            print(f"save_scan error: {e}")
            return None
        finally:
            cursor.close()

    def get_user_scans(self, user_id: int, limit: int = 10, offset: int = 0):
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT scan_id, original_text, prediction_result,
                       confidence_score, risk_level, scan_timestamp
                FROM email_scan
                WHERE user_id = %s
                ORDER BY scan_timestamp DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            )
            return cursor.fetchall()
        except Error as e:
            print(f"get_user_scans error: {e}")
            return []
        finally:
            cursor.close()

    def get_user_scan_stats(self, user_id: int):
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
            SELECT
                COUNT(*)  AS total_scans,
                SUM(CASE WHEN prediction_result = 'phishing' THEN 1 ELSE 0 END) AS phishing_count,
                SUM(CASE WHEN prediction_result = 'safe'     THEN 1 ELSE 0 END) AS safe_count,
                AVG(confidence_score) AS avg_confidence
                FROM email_scan
                WHERE user_id = %s
                """,
                (user_id,),
            )
            stats = cursor.fetchone()
            return {
                "total_scans": int(stats["total_scans"] or 0),
                "phishing_count": int(stats["phishing_count"] or 0),
                "safe_count": int(stats["safe_count"] or 0),
                "avg_confidence": float(stats["avg_confidence"] or 0.0),
            }
        except Error as e:
            print(f"get_user_scan_stats error: {e}")
            return {
                "total_scans": 0,
                "phishing_count": 0,
                "safe_count": 0,
                "avg_confidence": 0.0,
            }
        finally:
            cursor.close()

    def delete_scan(self, scan_id: int, user_id: int) -> bool:
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "DELETE FROM email_scan WHERE scan_id = %s AND user_id = %s",
                (scan_id, user_id),
            )
            self.connection.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                print(f"Scan {scan_id} deleted")
            else:
                print(f"Scan {scan_id} not found or unauthorized")
            return deleted
        except Error as e:
            self.connection.rollback()
            print(f"delete_scan error: {e}")
            return False
        finally:
            cursor.close()

    # FEEDBACK METHODS
    def save_feedback(
        self, user_id, scan_id, correct_prediction, rating, feedback_text, category
    ):
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            query = """
            INSERT INTO user_feedback
                (user_id, scan_id, correct_prediction, feedback_rating, feedback_text, reported_category, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(
                query,
                (user_id, scan_id, correct_prediction, rating, feedback_text, category),
            )
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            self.connection.rollback()
            print(f"save_feedback error: {e}")
            return None
        finally:
            cursor.close()

    def get_feedback_count(self) -> int:
        """Get total number of feedback entries"""
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM user_feedback")
            return cursor.fetchone()[0]
        except Error as e:
            print(f"get_feedback_count error: {e}")
            return 0
        finally:
            cursor.close()

    def get_feedback_by_scan_user(self, scan_id: int, user_id: int):
        """Check if feedback already exists for a scan by this user"""
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT feedback_id FROM user_feedback
                WHERE scan_id = %s AND user_id = %s
            """,
                (scan_id, user_id),
            )
            return cursor.fetchone()
        except Error as e:
            print(f"get_feedback_by_scan_user error: {e}")
            return None
        finally:
            cursor.close()

    def get_all_feedback(self, limit: int = 50, offset: int = 0) -> list:
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT f.*, u.user_name
                FROM user_feedback f
                LEFT JOIN users u ON f.user_id = u.user_id
                ORDER BY f.created_at DESC
                LIMIT %s OFFSET %s
            """,
                (limit, offset),
            )
            return cursor.fetchall()
        except Error as e:
            print(f"get_all_feedback error: {e}")
            return []
        finally:
            cursor.close()

    # SCAN BY ID (for feedback verification)
    def get_scan_by_id(self, scan_id: int):
        """Get a single scan by its ID"""
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT scan_id, user_id, original_text, processed_text,
                       prediction_result, confidence_score, risk_level, scan_timestamp
                FROM email_scan
                WHERE scan_id = %s
            """,
                (scan_id,),
            )
            return cursor.fetchone()
        except Error as e:
            print(f"get_scan_by_id error: {e}")
            return None
        finally:
            cursor.close()

    # PATTERNS
    def get_active_patterns(self):
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM phishing_patterns WHERE is_active = TRUE")
            return cursor.fetchall()
        except Error as e:
            print(f"get_active_patterns error: {e}")
            return []
        finally:
            cursor.close()

    def get_active_patterns_count(self, category: str = None) -> int:
        """Get total count of active patterns"""
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            if category:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM phishing_patterns 
                    WHERE is_active = TRUE AND category = %s
                """,
                    (category,),
                )
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM phishing_patterns 
                    WHERE is_active = TRUE
                """)
            return cursor.fetchone()[0]
        except Error as e:
            print(f"get_active_patterns_count error: {e}")
            return 0
        finally:
            cursor.close()

    def get_active_patterns_with_pagination(
        self, category: str = None, limit: int = 10, offset: int = 0
    ):
        """Get active phishing patterns with pagination"""
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            if category:
                query = """
                    SELECT * FROM phishing_patterns 
                    WHERE is_active = TRUE AND category = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(query, (category, limit, offset))
            else:
                query = """
                    SELECT * FROM phishing_patterns 
                    WHERE is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(query, (limit, offset))
            return cursor.fetchall()
        except Error as e:
            print(f"get_active_patterns error: {e}")
            return []
        finally:
            cursor.close()

    # ADMIN METHODS
    def get_admin_stats(self) -> dict:
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT
                (SELECT COUNT(*) FROM users) AS total_users,
                (SELECT COUNT(*) FROM email_scan) AS total_scans,
                (SELECT COUNT(*) FROM email_scan WHERE prediction_result = 'phishing') AS phishing_detected,
                (SELECT COUNT(*) FROM email_scan WHERE prediction_result = 'safe') AS safe_emails,
                (SELECT COUNT(*) FROM email_scan WHERE DATE(scan_timestamp) = CURDATE()) AS today_scans,
                (SELECT COUNT(DISTINCT user_id) FROM email_scan WHERE DATE(scan_timestamp) = CURDATE()) AS active_users_today
                """)
            row = cursor.fetchone()
            return {
                "total_users": int(row["total_users"] or 0),
                "total_scans": int(row["total_scans"] or 0),
                "phishing_detected": int(row["phishing_detected"] or 0),
                "safe_emails": int(row["safe_emails"] or 0),
                "today_scans": int(row["today_scans"] or 0),
                "active_users_today": int(row["active_users_today"] or 0),
            }
        except Error as e:
            print(f"get_admin_stats error: {e}")
            return {
                "total_users": 0,
                "total_scans": 0,
                "phishing_detected": 0,
                "safe_emails": 0,
                "today_scans": 0,
                "active_users_today": 0,
            }
        finally:
            cursor.close()

    def get_all_users(self, limit: int = 50, offset: int = 0) -> list:
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT user_id, user_name, email, created_at, last_login, total_scans, is_active
                FROM users
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            return cursor.fetchall()
        except Error as e:
            print(f"get_all_users error: {e}")
            return []
        finally:
            cursor.close()

    def get_users_count(self) -> int:
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM users")
            return cursor.fetchone()[0]
        except Error as e:
            print(f"get_users_count error: {e}")
            return 0
        finally:
            cursor.close()

    def get_all_scans(self, limit: int = 50, offset: int = 0) -> list:
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT e.scan_id, e.user_id, u.user_name, e.original_text,
                       e.prediction_result, e.confidence_score, e.risk_level,
                       e.scan_timestamp, e.processing_time
                FROM email_scan e
                LEFT JOIN users u ON e.user_id = u.user_id
                ORDER BY e.scan_timestamp DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            return cursor.fetchall()
        except Error as e:
            print(f"get_all_scans error: {e}")
            return []
        finally:
            cursor.close()

    def admin_delete_user(self, user_id: int) -> bool:
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            cursor.execute("DELETE FROM email_scan WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            self.connection.commit()
            return cursor.rowcount > 0
        except Error as e:
            self.connection.rollback()
            print(f"admin_delete_user error: {e}")
            return False
        finally:
            cursor.close()

    def get_analytics_data(self) -> dict:
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT DATE(scan_timestamp) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction_result='phishing' THEN 1 ELSE 0 END) as phishing,
                    SUM(CASE WHEN prediction_result='safe' THEN 1 ELSE 0 END) as safe
                FROM email_scan
                WHERE scan_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(scan_timestamp)
                ORDER BY date ASC
                """)
            daily_scans = cursor.fetchall()
            cursor.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM email_scan
                GROUP BY risk_level
                """)
            risk_dist = cursor.fetchall()

            return {"daily_scans": daily_scans, "risk_distribution": risk_dist}
        except Error as e:
            print(f"get_analytics_data error: {e}")
            return {"daily_scans": [], "risk_distribution": []}
        finally:
            cursor.close()

    # SHAP EXPLANATION
    def save_shap_explanation(self, scan_id: int, shap_data: dict) -> bool:
        """Save SHAP explanation data to database"""
        self.ensure_connection()
        cursor = self.connection.cursor()

        try:
            feature_names = shap_data.get("feature_names", [])
            shap_values = shap_data.get("shap_values", [])
            impacts = shap_data.get("impacts", [])
            strengths = shap_data.get("strengths", [])

            for i, feature_name in enumerate(feature_names):
                shap_value = shap_values[i] if i < len(shap_values) else 0
                impact = impacts[i] if i < len(impacts) else "positive"
                strength = strengths[i] if i < len(strengths) else "moderate"

                query = """
                    INSERT INTO shap_explanations 
                    (scan_id, feature_name, shap_value, impact, strength, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """
                cursor.execute(
                    query, (scan_id, feature_name, shap_value, impact, strength)
                )

            self.connection.commit()
            cursor.close()
            print(
                f"[DB] Saved {len(feature_names)} SHAP explanations for scan_id: {scan_id}"
            )
            return True

        except Exception as e:
            self.connection.rollback()
            print(f"[DB] Error saving SHAP explanation: {e}")
            cursor.close()
            return False

    # SYSTEM SETTINGS METHODS
    def get_all_settings(self) -> dict:
        """Get all system settings as dictionary"""
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT setting_key, setting_value, setting_type, category, description
                FROM system_settings
                ORDER BY category, setting_key
            """)
            results = cursor.fetchall()
            settings = {}
            for row in results:
                settings[row["setting_key"]] = {
                    "value": row["setting_value"],
                    "type": row["setting_type"],
                    "category": row["category"],
                    "description": row["description"],
                }
            return settings
        except Error as e:
            print(f"get_all_settings error: {e}")
            return {}
        finally:
            cursor.close()

    def get_setting(self, setting_key: str):
        """Get a single setting value"""
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT setting_value, setting_type, category, description
                FROM system_settings
                WHERE setting_key = %s
            """,
                (setting_key,),
            )
            return cursor.fetchone()
        except Error as e:
            print(f"get_setting error: {e}")
            return None
        finally:
            cursor.close()

    def update_setting(
        self, setting_key: str, setting_value: str, updated_by: int = None
    ) -> bool:
        """Update a system setting"""
        self.ensure_connection()
        cursor = self.connection.cursor()
        try:
            # First check if setting exists
            cursor.execute(
                "SELECT setting_id FROM system_settings WHERE setting_key = %s",
                (setting_key,),
            )
            exists = cursor.fetchone()

            if exists:
                query = """
                    UPDATE system_settings 
                    SET setting_value = %s, updated_by = %s, updated_at = NOW()
                    WHERE setting_key = %s
                """
                cursor.execute(query, (setting_value, updated_by, setting_key))
            else:
                # Insert new setting if doesn't exist
                query = """
                    INSERT INTO system_settings (setting_key, setting_value, updated_by, updated_at)
                    VALUES (%s, %s, %s, NOW())
                """
                cursor.execute(query, (setting_key, setting_value, updated_by))

            self.connection.commit()
            print(f"Setting '{setting_key}' updated to '{setting_value}'")
            return True
        except Error as e:
            self.connection.rollback()
            print(f"update_setting error: {e}")
            return False
        finally:
            cursor.close()

    def get_settings_by_category(self, category: str) -> dict:
        """Get all settings under a specific category"""
        self.ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT setting_key, setting_value, setting_type, description
                FROM system_settings
                WHERE category = %s
                ORDER BY setting_key
            """,
                (category,),
            )
            results = cursor.fetchall()
            settings = {}
            for row in results:
                settings[row["setting_key"]] = row["setting_value"]
            return settings
        except Error as e:
            print(f"get_settings_by_category error: {e}")
            return {}
        finally:
            cursor.close()


# checking connection
if __name__ == "__main__":
    db = Database()
    if db.connection and db.connection.is_connected():
        print("Database connection OKay...")
        db.close()
    else:
        print("Could not connect to database")
