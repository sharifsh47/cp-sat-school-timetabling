# Added typing for clarity and mypy friendliness
from typing import Dict, List, Optional, Set, Tuple, TypedDict, Union

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar, LinearExpr

 



#they make everything safer, clearer, and easier to run at scale. 
# they catch mistakes before the solver, let you swap datasets freely,
#  and keep your code stable as requirements change.
class RoomDict(TypedDict):
    id: str
    type: str


class ClassDict(TypedDict):
    id: str
    daily_max: int
    # Minimum lessons per day for the class
    daily_min: int


class TeacherDict(TypedDict):
    id: str
    quals: Set[str]
    unavailable: List[Tuple[int, int]]
    weekly_max: int


class CourseDict(TypedDict):
    class_id: str
    subject: str
    periods: int
    room_type: str

#ProblemDataDict = the checklist of everything the solver expects in the input.
#It doesn’t do magic at runtime; it makes your code safe, clean, and easy to use.
class ProblemDataDict(TypedDict):
    days: List[str]
    num_slots_per_day: int
    rooms: List[RoomDict]
    classes: List[ClassDict]
    teachers: List[TeacherDict]
    courses: List[CourseDict]
    # subjects that should prefer qualified teachers; empty/omitted means none
    specialist_subjects: Optional[List[str]]
    # subjects that should prefer double periods
    double_subjects: Optional[List[str]]
    # subjects considered "difficult" to cluster all on one day
    difficult_day_subjects: Optional[List[str]]
    # special per weekday maximum lessons (e.g., “Fridays ≤5”).
    per_day_max: Optional[Dict[str, int]]

# box of results you hand back after the solver finishes.
#total=False means the dictionary is not required to have all the keys.

class SolutionDict(TypedDict, total=False):
    status: str#"OPTIMAL", "FEASIBLE", "INFEASIBLE"
    objective_value: Optional[float]#The total penalty number
    class_timetables: Dict[str, Dict[str, List[str]]]#→ class id → day name → list of slots (strings like "Math:T_A@R101" or "-").
    # "C1": {"Mon": ["Math:T_A@R101", "German:T_A@R101", "-", "PE:T_A@Gym", "-", "-"],
    teacher_timetables: Dict[str, Dict[str, List[str]]]#→ teacher id → day name → list of slots (what they teach where).
    #"T_A": { "Mon": ["C1-Math@R101", "C1-German@R101", "-", "C1-PE@Gym", "-", "-"]
    stats: Dict[str, float]#  "stats": {"conflicts": 1234, "branches": 5678, "wall_time_s": 2.7},
    penalties: Dict[str, Dict[str, float]]# "penalties": { "unavailability": {"weight": 5, "count": 1, "weighted": 5},"double_period_singles": {"weight": 2, "count": 2, "weighted": 4},"total_weighted": {"weight": 1, "count": 0, "weighted": 12}
    num_slots_per_day: int
    run_config: Dict[str, Optional[Union[int, float]]]
def solve_timetabling_problem(
    problem_data: ProblemDataDict,
    *,
    # Soft penalty for assigning a teacher at a marked-unavailable slot
    unavailability_weight: int = 5,
    # Soft penalty for assigning an unqualified teacher (for specialist subjects)
    unqualified_weight: int = 3,
    time_limit_seconds: float = 20.0,#The solver keeps searching for a better schedule until time runs out.
    random_seed: Optional[int] = None,#Seed = repeatable random choices.
    triple_difficult_weight: int = 2,#Penalty per period above the difficult-subject threshold per day
    # Soft penalty for specials that are not in double periods
    double_period_weight: int = 2,
    # Soft penalty for exceeding per-subject daily cap (>2 per day per class)
    subject_daily_cap_weight: int = 2,
) -> SolutionDict:#The function returns a dictionary

    """Solve the school timetabling CP-SAT model.

    Hard constraints (must hold):
      • Each course scheduled exactly for its required periods.
      • No overlaps for classes, teachers, or rooms.
      • Per-class daily min/max bounds (plus optional per-day overrides).
      • Each class's lessons per day form one contiguous block.
      • First slot occupied each day (current setting).

    Soft constraints (penalized in objective):
      • Teacher assigned during an unavailable slot.
      • Unqualified teacher on specialist subjects (from input specialist_subjects).
      • Double subjects not placed as double periods.
      • More than 3 difficult subjects in a day for a class (penalize excess).
      • Subject appears more than twice in a day for a class (penalize excess).
      

    Returns:
    - A dictionary with status, objective value (if minimized), and timetables
    """

    # Config from input (or empty defaults)
    specialist_list: List[str] = list(problem_data.get("specialist_subjects") or [])
    specialist_set: Set[str] = set(specialist_list)
    double_subjects: Set[str] = set(problem_data.get("double_subjects") or [])
    difficult_subjects: List[str] = list(problem_data.get("difficult_day_subjects") or [])
    # Fixed threshold: combined difficult-subject periods per day must be <= 3 (no penalty),
    # penalty triggers when total > 3
    difficult_threshold: int = 3
#the endpoint receives the JSON, and problem_data/problem_data_dict holds those values for the solver to use.
#Those lines are just taking things out of the box so the solver can use them.
#ProblemDataDict is just a schema/checklist: 
# it helps catch missing fields early (e.g., you forgot courses) and makes the code clearer.
#Taking pieces out of that JSON

#Putting them into variables with clear names

#Preparing fast lookup structures for later constraints
    days: List[str] = problem_data["days"] 
    num_slots_per_day: int = problem_data["num_slots_per_day"]
    slots: List[int] = list(range(num_slots_per_day))
    rooms: List[RoomDict] = problem_data["rooms"]
    classes: List[ClassDict] = problem_data["classes"]
    teachers: List[TeacherDict] = problem_data["teachers"]
    teacher_unavailable_sets: List[Set[Tuple[int, int]]] = [
        {tuple(pair) for pair in t["unavailable"]}
        for t in teachers
    ]
    courses: List[CourseDict] = problem_data["courses"]
    per_day_max_map: Dict[str, int] = problem_data.get("per_day_max") or {}

    # Index helpers
    # Quick lookup maps: translate human IDs ("C1", "Gym") to compact integers (0,1,2…)
# This makes it easy and fast to loop over classes/rooms and build arrays of variables.
    class_index = {c["id"]: i for i, c in enumerate(classes)} # e.g., {"C1": 0, "C2": 1, ...}
    model = cp_model.CpModel()#CpModel() is the container that holds your whole math problem

    # Decision variables
    course_scheduled: Dict[Tuple[int, int, int], IntVar] = {}
    #course_scheduled[(5, 1, 3)] = 1 means course #5 happens on Tuesday (day=1) at slot 3.
    # Is course c_idx scheduled on day d at slot s? a Boolean IntVar (0 or 1) created later with model.NewBoolVar(...)
    #4-tuple = a single yes/no decision about (course, day, slot, teacher).
    teacher_assigned: Dict[Tuple[int, int, int, int], IntVar] = {}#If course c_idx happens at (day d, slot s), is teacher t the one teaching it? a Boolean IntVar (0 or 1) created later with model.NewBoolVar(...)
    #teacher_assigned[(5, 1, 3, 2)] = 1 means teacher #2 teaches course #5 on Tue slot 3
    room_assigned: Dict[Tuple[int, int, int, int], IntVar] = {}#If course c_idx happens at (day d, slot s), is room r the one used? a Boolean IntVar (0 or 1) created later with model.NewBoolVar(...)
# We pre-create empty lists (buckets) for every (resource, day, slot) combination.
# Later, when we create each decision variable, we append it into the right bucket.
# Then we can add constraints like AddAtMostOne(bucket) to prevent double-booking.

    # Think of this as making buckets for every (class, day, slot).
    #Key = (class_index, day_index, slot_index) → e.g. (0, 2, 4) means Class #0 on Wednesday, slot 4.
    # *that class* at *that time*. Later: model.AddAtMostOne(class_time_to_course_scheduled[(ci,d,s)]).
    #List[IntVar] as a bucket of boolean switches (each switch is an IntVar that can only be 0 or 1).
    #3-tuple = a key to a list of those decisions for one (teacher, day, slot), used to enforce “no double-booking.”
    class_time_to_course_scheduled: Dict[Tuple[int, int, int], List[IntVar]] = {

            (ci, d, s): []
            for ci in range(len(classes))
            for d in range(len(days))
            for s in slots
        }
    
    # For each (teacher_idx, day_idx, slot_idx), keep all teacher_assigned[...] vars that assign
    # *that teacher* to *that time*. Later: model.AddAtMostOne(teacher_time_to_teacher_assigned[(ti,d,s)]).
    teacher_time_to_teacher_assigned: Dict[Tuple[int, int, int], List[IntVar]] = {
        # teacher cant teach two courses at hte same time
            (ti, d, s): []
            for ti in range(len(teachers))
            for d in range(len(days))
            for s in slots
        }
    # For each (room_idx, day_idx, slot_idx), keep all room_assigned[...] vars that use
    # *that room* at *that time*. Later: model.AddAtMostOne(room_time_to_room_assigned[(ri,d,s)]).
    room_time_to_room_assigned: Dict[Tuple[int, int, int], List[IntVar]] = {
      
            (ri, d, s): []
            for ri in range(len(rooms))
            for d in range(len(days))
            for s in slots
        }

    # which rooms are OK for this lesson
    #cls_idx: which class this course belongs to
    #room_type:"GEN", "GYM", "ART"
    #enumerate(rooms) gives you (i, r) where i is the index (0,1,2,…) and r is the room dict.
    
    def rooms_feasible_for(room_type: str) -> List[int]:
        return [i for i, r in enumerate(rooms) if r["type"] == room_type]

    # identify the course, find its class, consider all teachers, and restrict to the rooms that make sense for the course’s room type
    for c_idx, course in enumerate(courses):#Go over every course.
        cls_idx = class_index[course["class_id"]]#Look up which class this course belongs to (e.g., "C1") and turn that class ID into a small integer
        cand_teachers = list(range(len(teachers)))#Make a list of all teacher indices [0, 1, 2, …].
        feasible_rooms = rooms_feasible_for(course["room_type"])#Make a list of all room indices that match this course's room type (e.g., "GEN", "GYM", "ART").
        for d in range(len(days)):
            for s in slots:
                if not cand_teachers or not feasible_rooms:
                    continue#If the list of candidate teachers is empty or the list of allowed rooms is empty,#we skip this
                #We create a yes/no switch
                course_scheduled[(c_idx, d, s)] = model.NewBoolVar(f"course_scheduled_c{c_idx}_d{d}_s{s}")
                #We then drop that switch into a bucket
                class_time_to_course_scheduled[(cls_idx, d, s)].append(course_scheduled[(c_idx, d, s)])
                for t in cand_teachers:
                    teacher_assigned[(c_idx, d, s, t)] = model.NewBoolVar(
                        f"teacher_assigned_c{c_idx}_d{d}_s{s}_t{t}")
                    teacher_time_to_teacher_assigned[(t, d, s)].append(teacher_assigned[(c_idx, d, s, t)])
                for r in feasible_rooms:
                    room_assigned[(c_idx, d, s, r)] = model.NewBoolVar(
                        f"room_assigned_c{c_idx}_d{d}_s{s}_r{r}")
                    room_time_to_room_assigned[(r, d, s)].append(room_assigned[(c_idx, d, s, r)])

    # Each course must be scheduled for its required number of periods
    for c_idx, course in enumerate(courses):
        times = [
            #Each course_scheduled[(c_idx,d,s)] is 1 if we place the course at that time, else 0.
            course_scheduled[(c_idx, d, s)] for d in range(len(days))
            for s in slots if (c_idx, d, s) in course_scheduled
        ]
        if not times:
            raise ValueError(
                f"No feasible (day,slot) for class {course['class_id']} "
                f"subject {course['subject']}. Check room type and teacher qualifications."
            )
        model.Add(sum(times) == course["periods"])

    # Link chosen time with exactly one teacher and one room
    for key, val in course_scheduled.items():
        c_idx, d, s = key  # unpack the tuple key
        course_var = val   # the BoolVar (0/1)
        #Build a list of 0/1 switches for teachers at this exact (course, day, slot).
        #If it exists, grab it and put it in the list teach_vars.
        teach_vars = [teacher_assigned[(c_idx, d, s, t)] for t in range(len(teachers)) if (c_idx, d, s, t) in teacher_assigned]
        #Build a list of 0/1 switches for rooms at this exact (course, day, slot).
        #If it exists, grab it and put it in the list room_vars.
        room_vars = [room_assigned[(c_idx, d, s, r)] for r in range(len(rooms)) if (c_idx, d, s, r) in room_assigned]
        #If course_var = 1 (course is happening then), the sum must be 1 → exactly one teacher is assigned.
        model.Add(sum(teach_vars) == course_var)
        model.Add(sum(room_vars) == course_var)

    # No overlaps for classes, teachers, rooms
    #for loop
    for lst in class_time_to_course_scheduled.values():
        if lst:
            model.AddAtMostOne(lst)
    for lst in teacher_time_to_teacher_assigned.values():
        if lst:
            model.AddAtMostOne(lst)
    for lst in room_time_to_room_assigned.values():
        if lst:
            model.AddAtMostOne(lst)

    # Daily min/max per class per day
    for cls_idx, cls in enumerate(classes):#{"id":"C1", ...}).
        class_daily_min_default: int = int(cls["daily_min"])  # now required
        for d in range(len(days)):
            day_name = days[d]#Go day by day (Mon, Tue, … as 0,1,…).
            daily_vars = [course_scheduled[(c_idx, d, s)]#Build a list of all 0/1 switches for this class on this day across all slots.
                          for c_idx, course in enumerate(courses)#Go over all courses.
                          if course["class_id"] == cls["id"]#Only look at courses for this class.
                          for s in slots
                          if (c_idx, d, s) in course_scheduled]
            if not daily_vars:
                continue
            # Hard max: class-specific daily_max
            model.Add(sum(daily_vars) <= cls["daily_max"])
            # Optional per-day max override (e.g., Friday cap)
            if day_name in per_day_max_map:
                model.Add(sum(daily_vars) <= int(per_day_max_map[day_name]))
            # Enforce class-specific daily minimum
            if class_daily_min_default > 0:
                model.Add(sum(daily_vars) >= class_daily_min_default)#at least 4 1s

    # Weekly max per teacher
    for t_idx, t in enumerate(teachers):
        weekly_vars = [teacher_assigned_var for (c_idx, d, s, tt), teacher_assigned_var in teacher_assigned.items() if tt == t_idx]
        if weekly_vars:
            model.Add(sum(weekly_vars) <= t["weekly_max"])

    # Hard constraint: For each class and day, lessons must be in a single
    # continuous block (no gaps where children would be unsupervised).
    class_occ: Dict[Tuple[int, int, int], IntVar] = {}
    for cls_idx, cls in enumerate(classes):
        for d in range(len(days)):
            # Occupancy per slot for this class/day (0/1)
            for s in slots:
                occ = model.NewBoolVar(f"occ_cls{cls_idx}_d{d}_s{s}")#“light switch” for “class has a lesson here”.
                lst = class_time_to_course_scheduled[(cls_idx, d, s)]

#If there are any possible course variables at this time, set occ equal to their sum.
#Because we already have AddAtMostOne(lst), the sum is either 0 or 1.
                if lst:
                    model.Add(occ == sum(lst))
                else:
                    model.Add(occ == 0)
                class_occ[(cls_idx, d, s)] = occ#Save that occ boolean in a dictionary for later use.

            start_vars: List[IntVar] = []#Make an empty list to collect “start-of-block” switches for each slot.
            for s in range(len(slots)):
               # start = 1 means:“A block of lessons starts at this slot.”
                start = model.NewBoolVar(f"start_cls{cls_idx}_d{d}_s{s}")
                #So if the day begins with a lesson → start = 1.
                if s == 0:
                    model.Add(start == class_occ[(cls_idx, d, 0)])
                else:
                    prev = class_occ[(cls_idx, d, s - 1)]
                    curr = class_occ[(cls_idx, d, s)]
                    # Start of block when occupancy rises from 0 to 1
                    model.Add(start <= curr)
                    model.Add(start + prev <= 1)
                    model.Add(start >= curr - prev)
                start_vars.append(start)

            # 3) At most one start → the occupied slots form a single contiguous block
            model.Add(sum(start_vars) <= 1)

            # 4) Hard rule: the first slot must be occupied every day
            model.Add(class_occ[(cls_idx, d, 0)] == 1)




  # Soft constraints

    penalty_terms: List[PenaltyExpr] = []
    # Track components for detailed breakdown
    unavailability_vars: List[IntVar] = []
    unqualified_vars: List[IntVar] = []
    # Soft: subject daily cap (per class, per day): penalize excess over 2
    subject_daily_excess_vars: List[IntVar] = []
    for cls_idx, cls in enumerate(classes):
        for d in range(len(days)):
            # gather all subjects offered for this class
            subjects_for_class = {c["subject"] for c in courses if c["class_id"] == cls["id"]}
            for subj in subjects_for_class:
                #Collect all slots where this subject is scheduled today
                #ys = [1, 0, 1, 0, 0, 0]
                ys = [
                    course_scheduled[(ci, d, s)]
                    for ci, c in enumerate(courses)
                    if c["class_id"] == cls["id"] and c["subject"] == subj
                    for s in slots
                    if (ci, d, s) in course_scheduled
                ]
                if not ys:
                    continue
                max_count = len(slots)
                #Count how many times the subject appears today
                #total_count = how many 1s are in ys
                total_count = model.NewIntVar(0, max_count, f"subj_total_cls{cls_idx}_d{d}_{subj}")
                model.Add(total_count == sum(ys))
                # excess = max(0, total_count - 2)
                diff = model.NewIntVar(-max_count, max_count, f"subj_diff_cls{cls_idx}_d{d}_{subj}")
                model.Add(diff == total_count - 2)
                excess = model.NewIntVar(0, max_count, f"subj_excess_cls{cls_idx}_d{d}_{subj}")
                #“Set excess equal to the larger of diff and 0.”
                model.AddMaxEquality(excess, [diff, model.NewConstant(0)])
                penalty_terms.append(subject_daily_cap_weight * excess)
                subject_daily_excess_vars.append(excess)


#Go through every possible teacher assignment in the whole model.
    for (c_idx, d, s, t_idx), a_var in teacher_assigned.items():  # Corrected syntax
        # Soft: avoid scheduling during a teacher's unavailable time
        #Loop over every teacher assignment decision variable.
        #a_var is a 0/1 switch that becomes 1 if we pick teacher t_idx to teach course c_idx at day d, slot s.
        #Is teacher t_idx unavailable on this day d and slot s
        if (d, s) in teacher_unavailable_sets[t_idx]:
            penalty_terms.append(unavailability_weight * a_var)
            unavailability_vars.append(a_var)

        # Soft: prefer specialists for selected subjects (PE, English, Art
       # If the subject is one of the specialist ones "Sport", "Englisch", "Kunst", "Musik", and this teacher doesn’t have that qualification, 
       # add another penalty (again only counts if a_var = 1).
        subj = courses[c_idx]["subject"]
        if subj in specialist_set and subj not in teachers[t_idx]["quals"]:
            penalty_terms.append(unqualified_weight * a_var)
            unqualified_vars.append(a_var)



    # Soft constraint: prefer selected subjects as double periods (consecutive pairs)
    #We’ll store a counter per course for “how many single special slots” it ends up with.
    double_period_singles_vars: List[IntVar] = []
#
    for c_idx, course in enumerate(courses):
        # Only subjects listed in double_subjects participate in this soft rule.
        is_double_subject = course["subject"] in double_subjects
        if not is_double_subject:
            continue

    # all feasible placements for this course (0/1 vars)
        times: List[IntVar] = [
            course_scheduled[(c_idx, d, s)]
            for d in range(len(days))
            for s in slots
            if (c_idx, d, s) in course_scheduled
        ]
        if not times:
            continue

    # build non-overlapping pair selection vars:
    # z[d,s] can be 1 only if scheduled at (d,s) AND (d,s+1), and pairs cannot overlap
        pair_select_vars: List[IntVar] = []
        for d in range(len(days)):
            day_pair_vars: List[IntVar] = []
            for s in range(num_slots_per_day - 1):
                k1 = (c_idx, d, s)
                k2 = (c_idx, d, s + 1)
                if k1 in course_scheduled and k2 in course_scheduled:
                    x = course_scheduled[k1]
                    y = course_scheduled[k2]
                    z = model.NewBoolVar(f"pairsel_c{c_idx}_d{d}_s{s}")
                    # z can be 1 only if both slots are scheduled
                    model.Add(z <= x)
                    model.Add(z <= y)
                    day_pair_vars.append(z)
            # enforce non-overlap of selected pairs within the day
            if day_pair_vars:
                for s in range(len(day_pair_vars) - 1):
                    #If a course is in slots 2–3 and 3–4, you must not count both.
                    #This forbids selecting overlapping pairs 
                    model.Add(day_pair_vars[s] + day_pair_vars[s + 1] <= 1)
                pair_select_vars.extend(day_pair_vars)

    # singles == total_slots - 2 * (#non-overlapping selected pairs)
    #Sport scheduled 5 times,Two double periods → 4 slots used,One slot left alone
        total_slots = sum(times)
        num_pairs   = sum(pair_select_vars)

        singles = model.NewIntVar(0, len(times), f"singles_c{c_idx}")
        model.Add(singles == total_slots - 2 * num_pairs)

        penalty_terms.append(double_period_weight * singles)
        double_period_singles_vars.append(singles)

    # Soft constraint: penalize when the combined count across all listed difficult
    # subjects exceeds 'difficult_threshold' in a single day for a class.
    subj_names = difficult_subjects
    triple_difficult_vars: List[IntVar] = []
    if subj_names:
        for cls_idx, cls in enumerate(classes):
            for d in range(len(days)):
                #ys = [1, 0, 1, 1, 0, 0]
                #“Is a difficult subject scheduled here for this class on this day?
                ys = [
                    course_scheduled[(ci, d, s)]
                    for ci, course in enumerate(courses)
                    if course["class_id"] == cls["id"] and course["subject"] in subj_names
                    for s in slots
                    if (ci, d, s) in course_scheduled
                ]
                max_count = len(slots)
                #total_count = how many 1s are in ys
                total_count = model.NewIntVar(0, max_count, f"diff_total_count_cls{cls_idx}_d{d}")
                if ys:
                    model.Add(total_count == sum(ys))
                else:
                    model.Add(total_count == 0)
                # excess = max(0, total_count - threshold) counts how many over the limit
                thr = max(0, difficult_threshold)
                diff = model.NewIntVar(-max_count, max_count, f"diff_minus_thr_cls{cls_idx}_d{d}")
                model.Add(diff == total_count - thr)
                #Keep only the part that exceeds the limit
                excess = model.NewIntVar(0, max_count, f"diff_excess_cls{cls_idx}_d{d}")
                model.AddMaxEquality(excess, [diff, model.NewConstant(0)])
                penalty_terms.append(triple_difficult_weight * excess)
                triple_difficult_vars.append(excess)

    if penalty_terms:
        model.Minimize(sum(penalty_terms))
#OR-Tools solver object that will actually search for a solution.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds

    # Thesis configuration: single-threaded search .
    solver.parameters.num_search_workers = 1
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
    if random_seed is not None:
        solver.parameters.random_seed = random_seed

        #status tells you how it went: e.g. OPTIMAL, FEASIBLE, INFEASIBLE, or UNKNOWN (time ran out with nothing)
    status = solver.Solve(model)

    solution: SolutionDict = {
        "status": solver.StatusName(status),
        "objective_value": (
            solver.ObjectiveValue()
            if (
                penalty_terms and
                status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            )
            else None
        ),
        "class_timetables": {},
        "teacher_timetables": {},
        "stats": {
            "conflicts": float(solver.NumConflicts()),#dead-ends 
            "branches": float(solver.NumBranches()),#decision points explored.possibilities
            "wall_time_s": float(solver.WallTime()),#seconds from start to finish of the solve
        },
        "penalties": {},
    }

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solution

    # Compute detailed penalty breakdown
    #
    unavailability_count = float(sum(solver.Value(v) for v in unavailability_vars))
    unqualified_count = float(sum(solver.Value(v) for v in unqualified_vars))
    double_period_single_slots = float(sum(solver.Value(v) for v in double_period_singles_vars))
    triple_difficult_count = float(sum(solver.Value(v) for v in triple_difficult_vars))
    subject_daily_excess_count = float(sum(solver.Value(v) for v in subject_daily_excess_vars))

    solution["penalties"] = {
        "unavailability": {
            "weight": float(unavailability_weight),
            "count": unavailability_count,
            "weighted": float(unavailability_weight) * unavailability_count,
        },
        "unqualified": {
            "weight": float(unqualified_weight),
            "count": unqualified_count,
            "weighted": float(unqualified_weight) * unqualified_count,
        },
        "double_period_singles": {
            "weight": float(double_period_weight),
            "count": double_period_single_slots,
            "weighted": float(double_period_weight) * double_period_single_slots,
        },
        "triple_difficult": {
            "weight": float(triple_difficult_weight),
            "count": triple_difficult_count,
            "weighted": float(triple_difficult_weight) * triple_difficult_count,
        },
        "subject_daily_cap": {
            "weight": float(subject_daily_cap_weight),
            "count": subject_daily_excess_count,
            "weighted": float(subject_daily_cap_weight) * subject_daily_excess_count,
        },
        "total_weighted": {
            "weight": 1.0,
            "count": 0.0,
            "weighted": (
                float(solution["objective_value"])
                if solution["objective_value"] is not None
                else 0.0
            ),
        },
    }

    def chosen_teacher(c_idx: int, d: int, s: int) -> Optional[int]:
        for t_idx in range(len(teachers)): # Changed t to t_idx
            key = (c_idx, d, s, t_idx)
            if key in teacher_assigned and solver.Value(teacher_assigned[key]) == 1:
                return t_idx
        return None

    def chosen_room(c_idx: int, d: int, s: int) -> Optional[int]:
        for r_idx in range(len(rooms)): # Changed r to r_idx
            key = (c_idx, d, s, r_idx)
            if key in room_assigned and solver.Value(room_assigned[key]) == 1:
                return r_idx
        return None

    for cls in classes:
        for d, day in enumerate(days):
            row = []
            for s in slots:
                placed = None
                for c_idx, course in enumerate(courses):
                    if course["class_id"] != cls["id"]:#skip courses that don’t belong to this class
                        continue
                    if ((c_idx, d, s) in course_scheduled and
                            solver.Value(course_scheduled[(c_idx, d, s)]) == 1):
                            #Find which teacher and which room got selected for this course at this time.
                        t_idx = chosen_teacher(c_idx, d, s)
                        r_idx = chosen_room(c_idx, d, s)
                        assert t_idx is not None and r_idx is not None, "Model invariant violated"
                        #Build the cell text like "English:T_B@R102".
                        placed = (f"{course['subject']}:"
                                  f"{teachers[t_idx]['id']}@"
                                  f"{rooms[r_idx]['id']}") # Corrected indexing
                        break
                row.append(placed or "-")
            class_id = cls["id"]
            #setdefault(class_id, {}) creates an empty dict for this class the first time we see it.
            solution["class_timetables"].setdefault(class_id, {})[day] = row

    for t_idx, t_dict in enumerate(teachers): # Changed t to t_idx, added t_dict for clarity
        solution["teacher_timetables"][t_dict["id"]] = {}
        for d, day in enumerate(days):
            row = []
            for s in slots:
                cell = "-"
                for c_idx, course in enumerate(courses):
                    key = (c_idx, d, s, t_idx)
                    if key in teacher_assigned and solver.Value(teacher_assigned[key]) == 1:
                        r_idx = chosen_room(c_idx, d, s)
                        assert r_idx is not None, "Model invariant violated"
                        cell = (f"{course['class_id']}-"
                                f"{course['subject']}@"
                                f"{rooms[r_idx]['id']}") # Corrected indexing
                        break
                row.append(cell)
            solution["teacher_timetables"][t_dict["id"]][day] = row

    return solution