import re
from pydantic import BaseModel, field_validator, EmailStr
from typing import Optional

class UserRegisterValidator(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if len(v) > 20:
            raise ValueError('Username must be less than 20 characters')
        if not re.match("^[a-zA-Z0-9_]+$", v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v

    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        if len(v) > 15:
            raise ValueError('Password too long')
        return v

class EmailAnalysisValidator(BaseModel):
    email_text: str
    user_id: Optional[int] = None

    @field_validator('email_text')
    def validate_email_text(cls, v):
        if len(v) < 10:
            raise ValueError('Email text too short (minimum 10 characters)')
        if len(v) > 10000:
            raise ValueError('Email text too long (maximum 10000 characters)')
        v = v.replace('<script>', '').replace('</script>', '')
        return v