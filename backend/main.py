import json
import os
from typing import Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .solver import solve_timetabling_problem
except ImportError:
    # Allow running from within the backend directory (no package parent)
    from solver import solve_timetabling_problem

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
    unavailable: List[Tuple[int, int]]
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
    double_subjects: Optional[List[str]] = None
    difficult_day_subjects: Optional[List[str]] = None
    weights: Optional[Dict[str, int]] = None
    per_day_max: Optional[Dict[str, int]] = None


@app.post("/solve")
async def solve_timetabling(
    problem_data: ProblemData,
    seed: Optional[int] = Query(default=None),
    time_limit: float = Query(default=20.0, gt=0),
    workers: Optional[int] = Query(default=None, ge=1),
):
    try:
        solution = solve_timetabling_problem(
            problem_data.model_dump(),
            time_limit_seconds=time_limit,
            random_seed=seed,
            
        )
        solution["num_slots_per_day"] = problem_data.num_slots_per_day
        solution["run_config"] = {
            "seed": seed,
            "time_limit_seconds": time_limit,
            "num_search_workers": workers,
        }
        return solution
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Solver Error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/")
async def read_root():
    return {"message": "Timetabling Solver API"}

@app.get("/example")
async def get_example(size: str):
    valid_sizes = {"small", "medium", "large"}
    if size not in valid_sizes:
        raise HTTPException(status_code=400, detail="Invalid size parameter.")

    path = os.path.join(os.path.dirname(__file__), "benchmarks", f"{size}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Example data for '{size}' not found.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)