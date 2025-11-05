import json
import os
from typing import Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .solver import ProblemDataDict, solve_timetabling_problem
except ImportError:
    # Allow running from within the backend directory (no package parent)
    from solver import ProblemDataDict, solve_timetabling_problem

app = FastAPI()

# CORS setup (simplified for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for input data
class Room(BaseModel):
    id: str
    type: str

class Class(BaseModel):
    id: str
    daily_max: int
    daily_min: int
    headteacher_id: Optional[str] = None

class Teacher(BaseModel):
    id: str
    quals: Set[str]
    unavailable: Set[Tuple[int, int]]
    preferred: Set[Tuple[int, int]]
    weekly_max: int

class Course(BaseModel):
    class_id: str
    subject: str
    periods: int
    room_type: str

class ProblemData(BaseModel):
    days: List[str]
    num_slots_per_day: int
    rooms: List[Room]
    classes: List[Class]
    teachers: List[Teacher]
    courses: List[Course]
    specialist_subjects: Optional[List[str]] = None
    special_room_types: Optional[List[str]] = None
    difficult_day_subjects: Optional[List[str]] = None
    weights: Optional[Dict[str, int]] = None
    per_day_max: Optional[Dict[str, int]] = None


@app.post("/solve")
async def solve_timetabling(problem_data: ProblemData):
    try:
        solution = solve_timetabling_problem(problem_data.model_dump())
        solution["num_slots_per_day"] = problem_data.num_slots_per_day
        return solution
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Solver Error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/")
async def read_root():
    return {"message": "Timetabling Solver API"}

@app.get("/example")
async def example_problem():
    path = os.path.join(os.path.dirname(__file__), "example.json")
    with open(path, "r") as f:
        return json.load(f)
