import os
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError, jwt
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from db_connection import Database
from ml_models import PhishingModel, SHAPExplainer as SHAPExplainer
from validators import UserRegisterValidator, EmailAnalysisValidator

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY not set in .env file! Server cannot start")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", 10080))  # 7 days
auth_scheme = OAuth2PasswordBearer(
    tokenUrl="token", auto_error=False
)  # auto_error=False for optional auth
limiter = Limiter(key_func=get_remote_address)

# FastAPI initialization
app = FastAPI(title="AI Phishing Detector API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()
ml_model = PhishingModel()
shap_explainer = SHAPExplainer(ml_model)

# =========================================================
# PYDANTIC MODELS
# =========================================================


class UserRegisterValidator(BaseModel):
    email: str
    username: str
    password: str
    confirm_password: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class AdminLogin(BaseModel):
    username: str
    password: str


class UpdateUserRequest(BaseModel):
    username: str


class FeedbackRequest(BaseModel):
    user_id: int
    scan_id: int
    correct_prediction: bool
    rating: int
    feedback_text: Optional[str] = None
    category: str


class EmailAnalysisValidator(BaseModel):
    email_text: str
    user_id: Optional[int] = None


class TokenData(BaseModel):
    user_id: Optional[int] = None


ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# JWT HELPERS


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    data_to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    data_to_encode.update({"exp": expire})
    return jwt.encode(data_to_encode, SECRET_KEY, algorithm=ALGORITHM)


# FIX 1: Fixed get_current_user to handle "sub" correctly
def get_current_user(token: str = Depends(auth_scheme)):
    """Get current user from JWT token"""
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    print(f"[DEBUG] Token received: {token[:50]}...")
    unauth_error = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"[DEBUG] Decoded payload: {payload}")

        # FIX: Get user_id from "sub" field
        user_id_str = payload.get("sub")
        if user_id_str is None:
            print("[DEBUG] No 'sub' in payload")
            raise unauth_error

        # Check if it's admin token
        if user_id_str == "admin":
            # Admin token
            user_token = {"user_id": 0, "role": "admin"}
            return user_token

        user_id = int(user_id_str)
        user_token = TokenData(user_id=user_id)

    except JWTError as e:
        print(f"[DEBUG] JWT Error: {e}")
        raise unauth_error

    user = db.get_user_by_id(user_token.user_id)
    if user is None:
        print(f"[DEBUG] User not found for ID: {user_token.user_id}")
        raise unauth_error

    print(f"[DEBUG] User found: {user['user_name']}")
    return user


# Optional current user (for endpoints that work with or without auth)
async def get_current_user_optional(token: Optional[str] = Depends(auth_scheme)):
    """Get current user if authenticated, otherwise return None"""
    if token is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None or user_id_str == "admin":
            return None
        user = db.get_user_by_id(int(user_id_str))
        return user
    except JWTError:
        return None


# =========================================================
# ADMIN AUTHENTICATION
# =========================================================


async def get_current_admin(token: str = Depends(auth_scheme)):
    """Verify admin token and return admin info"""
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        role: str = payload.get("role")
        username: str = payload.get("sub")

        if role is None or username is None:
            raise credentials_exception

        if role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        token_data = {"username": username, "role": role, "token": token}

    except JWTError:
        raise credentials_exception

    return token_data


# =========================================================
# ROOT
# =========================================================


@app.get("/")
def root():
    return {
        "message": "AI Phishing Detector API is running",
        "status": "ok",
        "ml_model": "loaded" if ml_model.model else "fallback",
    }


# =========================================================
# AUTH
# =========================================================


@app.post("/api/register")
@limiter.limit("20 per hour")
def register(request: Request, user: UserRegisterValidator):
    # Check if passwords match
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    email_exist = db.get_user_by_email(user.email)
    if email_exist:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = db.create_user(user.username, user.email, user.password)
    if user_id:
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "username": user.username,
            "email": user.email,
        }
    raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/auth/login")
@limiter.limit("30 per minute")
def login(request: Request, user: UserLogin):
    user_exist = db.get_user_by_email(user.email)
    if not user_exist or not db.verify_password(user.password, user_exist["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    db.update_last_login(user_exist["user_id"])

    # FIX: Use "sub" field for user_id (consistent with get_current_user)
    access_token = create_access_token(
        data={"sub": str(user_exist["user_id"])},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {
        "success": True,
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_exist["user_id"],
        "username": user_exist["user_name"],
        "email": user_exist["email"],
        "total_scans": user_exist["total_scans"],
    }


# =========================================================
# USER ENDPOINTS
# =========================================================


@app.get("/api/users/me")
def get_my_profile(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["user_id"],
        "username": current_user["user_name"],
        "email": current_user["email"],
        "total_scans": current_user["total_scans"],
        "created_at": str(current_user.get("created_at", "")),
        "last_login": str(current_user.get("last_login", "")),
    }


@app.put("/api/users/me")
@limiter.limit("3 per minute")
def update_user_me(
    request: Request,
    body: UpdateUserRequest,
    current_user: dict = Depends(get_current_user),
):
    if not body.username or len(body.username) < 3:
        raise HTTPException(
            status_code=400, detail="Username must be at least 3 characters"
        )

    db.ensure_connection()
    cursor = db.connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET user_name = %s WHERE user_id = %s",
            (body.username, current_user["user_id"]),
        )
        db.connection.commit()
        return {
            "success": True,
            "message": "Username updated",
            "username": body.username,
        }
    except Exception as e:
        db.connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()


# =========================================================
# EMAIL ANALYSIS
# =========================================================


def fallback_phishing_method(email_text: str):
    email_lower = email_text.lower()
    phishing_keywords = [
        "urgent",
        "click here",
        "verify",
        "password",
        "account suspended",
    ]
    matched = [kw for kw in phishing_keywords if kw in email_lower]

    if matched:
        return {
            "result": "phishing",
            "confidence": 60.0,
            "risk_level": "Medium",
            "risk_factors": matched,
        }
    else:
        return {
            "result": "safe",
            "confidence": 70.0,
            "risk_level": "Low",
            "risk_factors": [],
        }


# FIX 2: Updated analyze_email to save SHAP explanation
@app.post("/api/analyze")
def analyze_email(request: Request, email_to_check: EmailAnalysisValidator):
    try:
        start_time = time.time()
        pred_result = ml_model.predict(email_to_check.email_text)
        processing_time = int((time.time() - start_time) * 1000)
    except Exception as e:
        print(f"Model failed: {e}, using fallback")
        pred_result = fallback_phishing_method(email_to_check.email_text)
        processing_time = 0

    scan_id = None
    if email_to_check.user_id:
        processed_text = (
            ml_model.preprocess_text(email_to_check.email_text)
            if ml_model.model
            else email_to_check.email_text
        )

        scan_id = db.save_scan(
            user_id=email_to_check.user_id,
            original_text=email_to_check.email_text,
            processed_text=processed_text,
            prediction=pred_result["result"],
            confidence=pred_result["confidence"],
            risk_level=pred_result["risk_level"],
            processing_time=processing_time,
        )

        # NEW: Save SHAP explanation to database
        if scan_id and shap_explainer.explainer is not None:
            try:
                shap_result = shap_explainer.explain(
                    email_to_check.email_text, top_n=10
                )
                if shap_result and shap_result.get("using_shap", False):
                    shap_data = {
                        "feature_names": [
                            f["feature"] for f in shap_result.get("top_features", [])
                        ],
                        "shap_values": [
                            f["shap_value"] for f in shap_result.get("top_features", [])
                        ],
                        "impacts": [
                            f["impact"] for f in shap_result.get("top_features", [])
                        ],
                        "strengths": [
                            f["strength"] for f in shap_result.get("top_features", [])
                        ],
                    }
                    db.save_shap_explanation(scan_id, shap_data)
                    print(f"[SHAP] Saved explanation for scan_id: {scan_id}")
            except Exception as shap_err:
                print(f"[SHAP] Could not save explanation: {shap_err}")

    return {
        "success": True,
        "scan_id": scan_id,
        "result": pred_result["result"],
        "confidence": f"{pred_result['confidence']:.1f}%",
        "risk_level": pred_result["risk_level"],
        "risk_factors": pred_result["risk_factors"],
        "processing_time_ms": processing_time,
        "message": "Analysis completed successfully"
        + (" (fallback mode)" if ml_model.model is None else ""),
    }


# =========================================================
# SCAN HISTORY
# =========================================================


@app.get("/api/user/{user_id}/scans")
def get_user_scans(
    user_id: int,
    limit: int = 20,
    page: int = 1,
    current_user: dict = Depends(get_current_user),
):
    # FIX: Handle both user dict and admin token
    current_user_id = current_user.get("user_id")
    if current_user_id != user_id and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403, detail="You can only view your own scan history"
        )

    offset = (page - 1) * limit
    scans = db.get_user_scans(user_id, limit, offset)
    stats = db.get_user_scan_stats(user_id)

    return {
        "success": True,
        "scans": scans,
        "stats": stats,
        "page": page,
        "limit": limit,
        "total": stats.get("total_scans", 0),
    }


@app.delete("/api/user/{user_id}/scan/{scan_id}")
def delete_scan(
    user_id: int, scan_id: int, current_user: dict = Depends(get_current_user)
):
    if current_user["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="You can only delete your own scans"
        )

    deleted = db.delete_scan(scan_id, user_id)
    if deleted:
        return {"success": True, "message": "Scan deleted successfully"}
    raise HTTPException(status_code=404, detail="Scan not found or unauthorized")


# =========================================================
# FEEDBACK
# =========================================================


@app.post("/api/feedback")
@limiter.limit("10 per minute")
def submit_feedback(
    request: Request,
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    if current_user["user_id"] != feedback.user_id:
        raise HTTPException(
            status_code=404, detail="Cannot give feedback for other users"
        )

    scan = db.get_scan_by_id(feedback.scan_id)
    if not scan or scan["user_id"] != feedback.user_id:
        raise HTTPException(status_code=404, detail="Scan not found")

    existing = db.get_feedback_by_scan_user(feedback.scan_id, feedback.user_id)
    if existing:
        raise HTTPException(
            status_code=400, detail="Feedback already given for this scan"
        )

    feedback_id = db.save_feedback(
        user_id=feedback.user_id,
        scan_id=feedback.scan_id,
        correct_prediction=feedback.correct_prediction,
        rating=feedback.rating,
        feedback_text=feedback.feedback_text,
        category=feedback.category,
    )
    if feedback_id:
        return {"success": True, "feedback_id": feedback_id}

    raise HTTPException(status_code=500, detail="Failed to save feedback")


# =========================================================
# ADMIN FEEDBACK
# =========================================================


@app.get("/api/admin/feedback")
@limiter.limit("30 per minute")
def admin_get_feedback(
    request: Request,
    admin: dict = Depends(get_current_admin),
    limit: int = 100,
    page: int = 1,
):
    try:
        offset = (page - 1) * limit
        feedback = db.get_all_feedback(limit, offset)

        return {
            "success": True,
            "feedback": feedback,
            "page": page,
            "limit": limit,
            "total": len(feedback),
        }
    except Exception as e:
        print(f"Admin feedback error: {e}")
        return {"success": False, "error": str(e), "feedback": []}


# =========================================================
# STATISTICS
# =========================================================


@app.get("/api/user/{user_id}/stats")
def get_user_stats(
    request: Request, user_id: int, current_user: dict = Depends(get_current_user)
):
    # FIX: Handle both user dict and admin token
    current_user_id = current_user.get("user_id")
    if current_user_id != user_id and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403, detail="You can only view your own statistics"
        )

    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        stats = db.get_user_scan_stats(user_id)
        recent_scans = db.get_user_scans(user_id, limit=10, offset=0)
        if not stats:
            stats = {
                "total_scans": 0,
                "phishing_count": 0,
                "safe_count": 0,
                "avg_confidence": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
            }
        return {
            "success": True,
            "user_id": user_id,
            "username": user.get("user_name", ""),
            "stats": stats,
            "recent_activity": recent_scans if recent_scans else [],
            "total_recent": len(recent_scans) if recent_scans else 0,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch statistics: {str(e)}"
        )


# =========================================================
# PATTERNS
# =========================================================


@app.get("/api/patterns")
def get_phishing_patterns(
    category: Optional[str] = None, limit: int = 10, page: int = 1
):
    offset = (page - 1) * limit
    patterns = db.get_active_patterns(category, limit, offset)
    total = db.get_active_patterns_count(category)

    return {
        "success": True,
        "patterns": patterns,
        "page": page,
        "limit": limit,
        "total": total,
    }


# =========================================================
# HEALTH CHECK
# =========================================================


@app.get("/api/health")
def health_check():
    db_status = "disconnected"
    if db.connection:
        try:
            db.connection.ping(reconnect=True)
            db_status = "connected"
        except:
            db_status = "error"

    model_status = "loaded" if ml_model.model else "fallback"

    if db_status != "connected":
        overall_status = "degraded"
    elif model_status == "fallback":
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "ml_model": model_status,
    }


# =========================================================
# MODEL INFO
# =========================================================


@app.get("/api/model/info")
def get_model_info():
    return ml_model.get_model_info()


# =========================================================
# SHAP EXPLAINER
# =========================================================


@app.post("/api/explain")
def get_explanation(email_req: EmailAnalysisValidator):
    if shap_explainer.explainer is None:
        return {"explanation": "SHAP not available", "using_fallback": True}
    try:
        features = ml_model.extract_features(email_req.email_text)
        explanation = shap_explainer.explain(email_req.email_text, features)
        return {"success": True, "explanation": explanation, "using_shap": True}
    except Exception as e:
        return {"success": False, "error": str(e), "using_fallback": True}


# =========================================================
# ADMIN ENDPOINTS
# =========================================================


@app.post("/api/admin/login")
@limiter.limit("60 per minute")
def admin_login(request: Request, credentials: AdminLogin):
    if (
        credentials.username == ADMIN_USERNAME
        and credentials.password == ADMIN_PASSWORD
    ):
        token = create_access_token(
            data={"sub": "admin", "role": "admin"}, expires_delta=timedelta(minutes=720)
        )
        return {
            "success": True,
            "access_token": token,
            "token_type": "bearer",
            "role": "admin",
            "username": "admin",
        }
    raise HTTPException(status_code=401, detail="Invalid admin credentials")


@app.get("/api/admin/stats")
def admin_get_stats(admin: dict = Depends(get_current_admin)):
    try:
        stats = db.get_admin_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "total_users": 0,
                "total_scans": 0,
                "phishing_detected": 0,
                "safe_emails": 0,
                "today_scans": 0,
                "active_users_today": 0,
            },
        }


@app.get("/api/admin/users")
def admin_get_users(
    limit: int = 20, page: int = 1, admin: dict = Depends(get_current_admin)
):
    try:
        offset = (page - 1) * limit
        users = db.get_all_users(limit, offset)
        total_users = db.get_users_count()
        return {
            "success": True,
            "users": users,
            "total": total_users,
            "page": page,
            "limit": limit,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "users": [], "total": 0}


@app.get("/api/admin/scans")
def admin_get_scans(
    limit: int = 20, page: int = 1, admin: dict = Depends(get_current_admin)
):
    try:
        offset = (page - 1) * limit
        scans = db.get_all_scans(limit, offset)
        total_stats = db.get_admin_stats()
        return {
            "success": True,
            "scans": scans,
            "total": total_stats.get("total_scans", 0),
            "page": page,
            "limit": limit,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "scans": [], "total": 0}


@app.delete("/api/admin/users/{user_id}")
@limiter.limit("20 per minute")
def admin_delete_user(
    request: Request, user_id: int, admin: dict = Depends(get_current_admin)
):
    try:
        success = db.admin_delete_user(user_id)
        if success:
            return {"success": True, "message": f"User {user_id} deleted"}
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics")
def admin_get_analytics(admin: dict = Depends(get_current_admin)):
    try:
        analytics = db.get_analytics_data()
        return {"success": True, "analytics": analytics}
    except Exception as e:
        return {"success": False, "error": str(e), "analytics": {}}
# =========================================================
# SYSTEM SETTINGS ENDPOINTS (FIXED)
# =========================================================

@app.get("/api/admin/settings")
@limiter.limit("30 per minute")
def get_all_settings(request: Request, admin: dict = Depends(get_current_admin)):
    """Get all system settings (admin only)"""
    try:
        settings = db.get_all_settings()
        return {
            "success": True,
            "settings": settings
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "settings": {}
        }


@app.get("/api/admin/settings/{setting_key}")
@limiter.limit("30 per minute")
def get_setting(request: Request, setting_key: str, admin: dict = Depends(get_current_admin)):
    """Get a specific setting value (admin only)"""
    try:
        setting = db.get_setting(setting_key)
        if setting:
            return {
                "success": True,
                "setting": {
                    "key": setting_key,
                    "value": setting['setting_value'],
                    "type": setting.get('setting_type', 'text'),
                    "category": setting.get('category', 'general'),
                    "description": setting.get('description', '')
                }
            }
        else:
            raise HTTPException(status_code=404, detail=f"Setting '{setting_key}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/settings/{setting_key}")
@limiter.limit("20 per minute")
def update_setting(
    request: Request,
    setting_key: str,
    setting_value: str,
    admin: dict = Depends(get_current_admin)
):
    """Update a system setting (admin only)"""
    try:
        admin_id = admin.get('admin_id', 1)
        
        success = db.update_setting(setting_key, setting_value, admin_id)
        if success:
            return {
                "success": True,
                "message": f"Setting '{setting_key}' updated successfully",
                "setting_key": setting_key,
                "setting_value": setting_value
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to update setting '{setting_key}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/settings/category/{category}")
@limiter.limit("30 per minute")
def get_settings_by_category(request: Request, category: str, admin: dict = Depends(get_current_admin)):
    """Get all settings under a specific category (admin only)"""
    try:
        settings = db.get_settings_by_category(category)
        return {
            "success": True,
            "category": category,
            "settings": settings
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "settings": {}
        }
# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting AI Phishing Detector API...")
    print(f"ML Model: {'Loaded' if ml_model.model else 'Fallback'}")
    print("Security Features: Rate Limiting + JWT + bcrypt + Input Validation")
    print("http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
