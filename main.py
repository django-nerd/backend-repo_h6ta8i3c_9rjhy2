import os
from datetime import datetime, timezone
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from database import db

app = FastAPI(title="InvestEd API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Utility helpers
# -------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_object_id(id_str: str):
    from bson import ObjectId
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")


def serialize_doc(doc: dict):
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    # Convert datetimes to iso
    for k, v in list(doc.items()):
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


# -------------------------------
# Auth & Users
# -------------------------------
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Literal["learner", "investor"]
    age: int = Field(..., ge=3, le=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserProfile(BaseModel):
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = ""


class User(BaseModel):
    name: str
    email: EmailStr
    role: Literal["learner", "investor"]
    age: int
    profile: UserProfile = Field(default_factory=UserProfile)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def hash_password(pw: str) -> str:
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest()


@app.post("/auth/register")
def register(payload: RegisterRequest):
    # Basic age gate: investors must be 18+
    if payload.role == "investor" and payload.age < 18:
        raise HTTPException(status_code=400, detail="Investors must be 18+")

    existing = db.user.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_doc = {
        "name": payload.name,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "role": payload.role,
        "age": payload.age,
        "profile": UserProfile().model_dump(),
        "kyc": {
            "status": "pending",
            "documents": [],
        },
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db.user.insert_one(user_doc)
    created = db.user.find_one({"_id": res.inserted_id})
    doc = serialize_doc(created)
    doc.pop("password_hash", None)
    return {"user": doc}


@app.post("/auth/login")
def login(payload: LoginRequest):
    user = db.user.find_one({"email": payload.email})
    if not user or user.get("password_hash") != hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    doc = serialize_doc(user)
    doc.pop("password_hash", None)
    return {"user": doc}


@app.delete("/auth/delete/{user_id}")
def delete_account(user_id: str):
    _id = to_object_id(user_id)
    db.user.delete_one({"_id": _id})
    # Cascade minimal cleanup
    db.investment.delete_many({"investor_id": user_id})
    db.payment_method.delete_many({"user_id": user_id})
    return {"status": "deleted"}


# -------------------------------
# KYC stub endpoints (store metadata only)
# -------------------------------
class KYCDocument(BaseModel):
    type: Literal["passport", "driver_license", "proof_of_address", "selfie"]
    url: str


class KYCUpdate(BaseModel):
    documents: List[KYCDocument]


@app.post("/kyc/submit/{user_id}")
def submit_kyc(user_id: str, payload: KYCUpdate):
    db.user.update_one(
        {"_id": to_object_id(user_id)},
        {
            "$set": {"kyc.status": "under_review", "updated_at": datetime.now(timezone.utc)},
            "$push": {"kyc.documents": {"$each": [d.model_dump() for d in payload.documents]}},
        },
    )
    return {"status": "received"}


# -------------------------------
# Learner Applications & Profiles
# -------------------------------
class Milestone(BaseModel):
    title: str
    target_date: Optional[str] = None
    progress: int = Field(0, ge=0, le=100)


class Certification(BaseModel):
    title: str
    issuer: Optional[str] = None
    verified: bool = False


class Testimonial(BaseModel):
    author: str
    text: str
    created_at: Optional[str] = None


class LearnerApplication(BaseModel):
    user_id: str
    name: str
    age: int = Field(..., ge=3, le=120)
    skills: List[str] = []
    field_of_study: Optional[str] = None
    project_description: str
    requested_funding: float = Field(..., gt=0)
    investment_terms: Optional[str] = ""
    return_model: Literal["ISA", "Revenue", "Bonus", "Hybrid"] = "ISA"
    payment_setup_done: bool = False


@app.post("/learners/apply")
def apply_learner(app: LearnerApplication):
    if app.age < 3:
        raise HTTPException(status_code=400, detail="Invalid age")
    doc = app.model_dump()
    doc.update(
        {
            "funded_amount": 0.0,
            "progress": 0,
            "milestones": [],
            "certifications": [],
            "testimonials": [],
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    res = db.learner.insert_one(doc)
    created = db.learner.find_one({"_id": res.inserted_id})
    return {"learner": serialize_doc(created)}


@app.get("/learners/explore")
def explore_learners(q: Optional[str] = None, limit: int = Query(50, le=100)):
    filt = {"status": {"$in": ["approved", "pending"]}, "$expr": {"$lt": ["$funded_amount", "$requested_funding"]}}
    if q:
        filt.update({"$or": [{"name": {"$regex": q, "$options": "i"}}, {"skills": {"$elemMatch": {"$regex": q, "$options": "i"}}}]})
    cursor = db.learner.find(filt).limit(limit)
    return {"learners": [serialize_doc(x) for x in cursor]}


@app.get("/learners/{learner_id}")
def get_learner(learner_id: str):
    doc = db.learner.find_one({"_id": to_object_id(learner_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"learner": serialize_doc(doc)}


# -------------------------------
# Investments & Payments (mocked processor)
# -------------------------------
class InvestmentRequest(BaseModel):
    investor_id: str
    learner_id: str
    amount: float = Field(..., gt=0)
    model: Literal["ISA", "Revenue", "Bonus", "Hybrid"]


@app.post("/investments")
def create_investment(req: InvestmentRequest):
    learner = db.learner.find_one({"_id": to_object_id(req.learner_id)})
    if not learner:
        raise HTTPException(status_code=404, detail="Learner not found")

    remaining = float(learner["requested_funding"]) - float(learner.get("funded_amount", 0))
    if req.amount > remaining + 1e-6:
        raise HTTPException(status_code=400, detail=f"Max allowed {remaining:.2f}")

    inv = {
        "investor_id": req.investor_id,
        "learner_id": req.learner_id,
        "amount": req.amount,
        "model": req.model,
        "created_at": datetime.now(timezone.utc),
        "status": "completed",
    }
    res = db.investment.insert_one(inv)
    # update learner
    new_funded = float(learner.get("funded_amount", 0)) + req.amount
    updates = {"funded_amount": new_funded, "updated_at": datetime.now(timezone.utc)}
    if new_funded + 1e-6 >= float(learner["requested_funding"]):
        updates["status"] = "funded"
    db.learner.update_one({"_id": learner["_id"]}, {"$set": updates})

    # create notification for learner
    notif = {
        "user_id": learner["user_id"],
        "title": "New investment received",
        "message": f"You received ${req.amount:.2f}",
        "type": "success",
        "is_read": False,
        "created_at": datetime.now(timezone.utc),
    }
    db.notification.insert_one(notif)

    created = db.investment.find_one({"_id": res.inserted_id})
    return {"investment": serialize_doc(created)}


@app.get("/investments/portfolio/{investor_id}")
def get_portfolio(investor_id: str):
    invs = list(db.investment.find({"investor_id": investor_id}))
    total_invested = sum(float(i.get("amount", 0)) for i in invs)
    # Demo ROI calculation: 6% annualized prorated by age in days
    roi = round(total_invested * 0.06, 2)
    investments = [serialize_doc(i) for i in invs]
    return {
        "summary": {
            "total_invested": round(total_invested, 2),
            "roi": round(roi, 2),
            "upcoming_payments": round(total_invested * 0.02, 2),
        },
        "investments": investments,
    }


# Payment methods (mock storage)
class PaymentMethod(BaseModel):
    user_id: str
    type: Literal["card", "bank", "digital"]
    details: dict
    confirmed: bool = False


@app.post("/payments/methods")
def add_payment_method(pm: PaymentMethod):
    doc = pm.model_dump()
    doc.update({"created_at": datetime.now(timezone.utc)})
    res = db.payment_method.insert_one(doc)
    created = db.payment_method.find_one({"_id": res.inserted_id})
    return {"payment_method": serialize_doc(created)}


@app.get("/payments/methods/{user_id}")
def list_payment_methods(user_id: str):
    pms = list(db.payment_method.find({"user_id": user_id}))
    return {"payment_methods": [serialize_doc(x) for x in pms]}


@app.delete("/payments/methods/{pm_id}")
def delete_payment_method(pm_id: str):
    db.payment_method.delete_one({"_id": to_object_id(pm_id)})
    return {"status": "deleted"}


@app.post("/payments/methods/confirm/{pm_id}")
def confirm_payment_method(pm_id: str):
    db.payment_method.update_one({"_id": to_object_id(pm_id)}, {"$set": {"confirmed": True}})
    return {"status": "confirmed"}


# -------------------------------
# Forum
# -------------------------------
class PostCreate(BaseModel):
    author_id: str
    title: str
    content: str


class ReplyCreate(BaseModel):
    author_id: str
    content: str


@app.post("/forum/posts")
def create_post(post: PostCreate):
    doc = {
        "author_id": post.author_id,
        "title": post.title,
        "content": post.content,
        "likes": [],  # store user_ids
        "views": 0,
        "created_at": datetime.now(timezone.utc),
    }
    res = db.post.insert_one(doc)
    created = db.post.find_one({"_id": res.inserted_id})
    return {"post": serialize_doc(created)}


@app.get("/forum/posts")
def list_posts(limit: int = Query(50, le=100)):
    cursor = db.post.find({}).sort("created_at", -1).limit(limit)
    posts = []
    for p in cursor:
        s = serialize_doc(p)
        s["like_count"] = len(s.get("likes", []))
        posts.append(s)
    return {"posts": posts}


@app.get("/forum/posts/{post_id}")
def get_post(post_id: str):
    db.post.update_one({"_id": to_object_id(post_id)}, {"$inc": {"views": 1}})
    p = db.post.find_one({"_id": to_object_id(post_id)})
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    replies = list(db.reply.find({"post_id": post_id}).sort("created_at", 1))
    return {"post": serialize_doc(p), "replies": [serialize_doc(r) for r in replies]}


@app.post("/forum/posts/{post_id}/like")
def like_post(post_id: str, user_id: str):
    p = db.post.find_one({"_id": to_object_id(post_id)})
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    likes = set([*p.get("likes", [])])
    likes.add(user_id)
    db.post.update_one({"_id": p["_id"]}, {"$set": {"likes": list(likes)}})
    return {"like_count": len(likes)}


@app.delete("/forum/posts/{post_id}")
def delete_post(post_id: str):
    db.post.delete_one({"_id": to_object_id(post_id)})
    db.reply.delete_many({"post_id": post_id})
    return {"status": "deleted"}


@app.post("/forum/posts/{post_id}/replies")
def add_reply(post_id: str, reply: ReplyCreate):
    if not db.post.find_one({"_id": to_object_id(post_id)}):
        raise HTTPException(status_code=404, detail="Post not found")
    doc = {
        "post_id": post_id,
        "author_id": reply.author_id,
        "content": reply.content,
        "created_at": datetime.now(timezone.utc),
    }
    res = db.reply.insert_one(doc)
    created = db.reply.find_one({"_id": res.inserted_id})
    return {"reply": serialize_doc(created)}


# -------------------------------
# Notifications & Recommendations
# -------------------------------
@app.get("/notifications/{user_id}")
def list_notifications(user_id: str):
    cur = db.notification.find({"user_id": user_id}).sort("created_at", -1)
    return {"notifications": [serialize_doc(x) for x in cur]}


@app.get("/recommendations/{investor_id}")
def recommend_learners(investor_id: str, limit: int = 5):
    # Simple heuristic: latest learners needing funds
    cur = db.learner.find({"$expr": {"$lt": ["$funded_amount", "$requested_funding"]}}).sort("created_at", -1).limit(limit)
    return {"learners": [serialize_doc(x) for x in cur]}


# -------------------------------
# Static/utility endpoints
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "InvestEd Backend Running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
        else:
            response["database"] = "❌ Not Initialized"
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:80]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
