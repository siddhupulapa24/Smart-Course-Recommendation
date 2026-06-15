"""
=============================================================
  SMART COURSE RECOMMENDATION LOGIC
  Rule-Based + Probabilistic Reasoning System
  No ML / Black-box models — Fully Transparent & Explainable
=============================================================
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Course:
    code: str
    name: str
    credits: int
    difficulty: int          # 1 (easy) → 5 (very hard)
    prerequisites: list[str] = field(default_factory=list)
    corequisites: list[str]  = field(default_factory=list)
    domain: str = ""
    description: str = ""


@dataclass
class StudentProfile:
    student_id: str
    name: str
    completed_courses: list[str]           # course codes already passed
    grades: dict[str, float]               # code → grade (0.0–10.0)
    current_semester: int                  # 1-based
    interests: list[str]                   # domains of interest
    workload_preference: str = "medium"    # light | medium | heavy


# ─────────────────────────────────────────────
#  COURSE DATABASE  (Prerequisites Rules DB)
# ─────────────────────────────────────────────

COURSE_DB: dict[str, Course] = {
    # ── Foundation
    "CS101": Course("CS101", "Introduction to Programming", 3, 1, [], [], "programming",
                    "Basics of programming using Python."),
    "MATH101": Course("MATH101", "Discrete Mathematics", 3, 2, [], [], "mathematics",
                      "Logic, sets, relations, graph theory."),
    "MATH102": Course("MATH102", "Linear Algebra", 3, 2, [], [], "mathematics",
                      "Vectors, matrices, eigenvalues."),
    "STAT101": Course("STAT101", "Probability & Statistics", 3, 2, ["MATH101"], [], "mathematics",
                      "Probability theory, distributions, inference."),

    # ── Intermediate
    "CS201": Course("CS201", "Data Structures", 3, 3, ["CS101", "MATH101"], [], "programming",
                    "Arrays, trees, graphs, hashing."),
    "CS202": Course("CS202", "Algorithms", 4, 3, ["CS201", "MATH101"], [], "programming",
                    "Sorting, dynamic programming, complexity."),
    "CS203": Course("CS203", "Database Systems", 3, 3, ["CS101"], [], "databases",
                    "Relational model, SQL, normalization."),
    "CS204": Course("CS204", "Operating Systems", 3, 4, ["CS201"], [], "systems",
                    "Processes, memory management, file systems."),

    # ── AI / ML Track
    "AI301": Course("AI301", "Artificial Intelligence", 4, 4,
                    ["CS202", "STAT101"], [], "ai",
                    "Search, knowledge representation, reasoning."),
    "AI302": Course("AI302", "Machine Learning", 4, 4,
                    ["AI301", "MATH102", "STAT101"], [], "ai",
                    "Supervised, unsupervised learning, evaluation."),
    "AI303": Course("AI303", "Natural Language Processing", 3, 4,
                    ["AI302"], [], "ai",
                    "Text processing, transformers, sentiment analysis."),
    "AI304": Course("AI304", "Knowledge Representation & Reasoning", 3, 4,
                    ["AI301", "MATH101"], [], "ai",
                    "Logic, ontologies, probabilistic reasoning."),

    # ── Data Science Track
    "DS301": Course("DS301", "Data Science Fundamentals", 3, 3,
                    ["STAT101", "CS201"], [], "data_science",
                    "EDA, feature engineering, pipelines."),
    "DS302": Course("DS302", "Big Data Analytics", 3, 4,
                    ["DS301", "CS203"], [], "data_science",
                    "Spark, Hadoop, distributed computing."),

    # ── Advanced
    "CS401": Course("CS401", "Advanced Algorithms", 4, 5,
                    ["CS202", "MATH102"], [], "programming",
                    "NP-completeness, approximation algorithms."),
    "CS402": Course("CS402", "Compiler Design", 3, 5,
                    ["CS202", "CS204"], [], "systems",
                    "Lexing, parsing, code generation."),
}


# ─────────────────────────────────────────────
#  RULE ENGINE  (Prerequisite Logic)
# ─────────────────────────────────────────────

class RuleEngine:
    """
    Hard rules — a recommendation is BLOCKED if any rule fires.
    These are deterministic, fully explainable checks.
    """

    def __init__(self, course_db: dict[str, Course]):
        self.db = course_db

    # ── Rule 1: All prerequisites must be completed
    def check_prerequisites(self, course: Course, student: StudentProfile) -> tuple[bool, list[str]]:
        missing = [p for p in course.prerequisites if p not in student.completed_courses]
        if missing:
            names = [self.db[c].name if c in self.db else c for c in missing]
            return False, [f"Missing prerequisite(s): {', '.join(names)}"]
        return True, []

    # ── Rule 2: Cannot re-take an already-completed course
    def check_not_already_done(self, course: Course, student: StudentProfile) -> tuple[bool, list[str]]:
        if course.code in student.completed_courses:
            return False, [f"Already completed '{course.name}'"]
        return True, []

    # ── Rule 3: Difficulty guard — don't jump more than 2 difficulty levels
    def check_difficulty_gap(self, course: Course, student: StudentProfile) -> tuple[bool, list[str]]:
        if not student.completed_courses:
            max_completed_diff = 0
        else:
            max_completed_diff = max(
                self.db[c].difficulty for c in student.completed_courses if c in self.db
            )
        gap = course.difficulty - max_completed_diff
        if gap > 2:
            return False, [
                f"Difficulty gap too large (course level {course.difficulty}, "
                f"your current level ~{max_completed_diff})"
            ]
        return True, []

    # ── Rule 4: Corequisites must be in current or completed courses
    def check_corequisites(self, course: Course,
                            student: StudentProfile,
                            enrolling_alongside: list[str]) -> tuple[bool, list[str]]:
        available = set(student.completed_courses) | set(enrolling_alongside)
        missing_co = [c for c in course.corequisites if c not in available]
        if missing_co:
            names = [self.db[c].name if c in self.db else c for c in missing_co]
            return False, [f"Missing co-requisite(s): {', '.join(names)}"]
        return True, []

    def evaluate(self, course: Course, student: StudentProfile,
                 enrolling_alongside: list[str] = None) -> tuple[bool, list[str]]:
        """Run all hard rules. Returns (eligible, list_of_blocking_reasons)."""
        enrolling_alongside = enrolling_alongside or []
        reasons = []
        for check_fn in [
            lambda: self.check_not_already_done(course, student),
            lambda: self.check_prerequisites(course, student),
            lambda: self.check_difficulty_gap(course, student),
            lambda: self.check_corequisites(course, student, enrolling_alongside),
        ]:
            ok, msgs = check_fn()
            if not ok:
                reasons.extend(msgs)

        eligible = len(reasons) == 0
        return eligible, reasons


# ─────────────────────────────────────────────
#  PROBABILISTIC REASONING MODULE
# ─────────────────────────────────────────────

class ProbabilisticReasoner:
    """
    Computes a confidence score P(success | student, course) ∈ [0, 1].

    Uses a transparent weighted formula — no black-box model:
        score = w1*grade_factor + w2*interest_factor + w3*difficulty_factor
                + w4*workload_factor
    Each factor is documented and explainable.
    """

    WEIGHTS = {
        "grade":       0.40,   # Prior performance in prereqs matters most
        "interest":    0.25,   # Domain alignment
        "difficulty":  0.20,   # Difficulty vs student level
        "workload":    0.15,   # Workload preference fit
    }

    def __init__(self, course_db: dict[str, Course]):
        self.db = course_db

    # ── Factor 1: Average grade in completed prerequisites (normalised 0→1)
    def _grade_factor(self, course: Course, student: StudentProfile) -> tuple[float, str]:
        prereq_grades = [
            student.grades[p] for p in course.prerequisites if p in student.grades
        ]
        if not prereq_grades:
            # No grade data → neutral assumption (Bayesian prior = 0.65)
            return 0.65, "No prior grade data; using neutral prior (0.65)"
        avg = sum(prereq_grades) / len(prereq_grades)
        normalised = avg / 10.0
        return normalised, f"Avg prerequisite grade {avg:.1f}/10 → factor {normalised:.2f}"

    # ── Factor 2: Interest alignment
    def _interest_factor(self, course: Course, student: StudentProfile) -> tuple[float, str]:
        if course.domain in student.interests:
            return 1.0, f"Domain '{course.domain}' matches student interests"
        return 0.4, f"Domain '{course.domain}' not in student interests"

    # ── Factor 3: Difficulty fit (penalise courses far above student's level)
    def _difficulty_factor(self, course: Course, student: StudentProfile) -> tuple[float, str]:
        if not student.completed_courses:
            completed_avg_diff = 1
        else:
            diffs = [self.db[c].difficulty for c in student.completed_courses if c in self.db]
            completed_avg_diff = sum(diffs) / len(diffs) if diffs else 1

        gap = course.difficulty - completed_avg_diff
        # sigmoid-like penalty: comfortable range → near 1.0; stretch → drops
        try:
            factor = 1 / (1 + math.exp(gap - 0.5))
        except OverflowError:
            factor = 0.0  # gap is extreme → near-zero confidence
        return factor, (f"Course difficulty {course.difficulty}, "
                        f"your avg {completed_avg_diff:.1f} → factor {factor:.2f}")

    # ── Factor 4: Workload preference
    def _workload_factor(self, course: Course, student: StudentProfile) -> tuple[float, str]:
        prefs = {"light": (1, 2), "medium": (2, 3), "heavy": (4, 5)}
        lo, hi = prefs.get(student.workload_preference, (2, 3))
        if lo <= course.difficulty <= hi:
            return 1.0, f"Workload matches preference '{student.workload_preference}'"
        dist = min(abs(course.difficulty - lo), abs(course.difficulty - hi))
        factor = max(0.3, 1.0 - 0.2 * dist)
        return factor, f"Workload mismatch (distance {dist}) → factor {factor:.2f}"

    def compute(self, course: Course, student: StudentProfile) -> dict:
        """Returns confidence score and full breakdown for transparency."""
        w = self.WEIGHTS
        gf, g_reason = self._grade_factor(course, student)
        inf, i_reason = self._interest_factor(course, student)
        df, d_reason = self._difficulty_factor(course, student)
        wf, w_reason = self._workload_factor(course, student)

        score = (w["grade"] * gf +
                 w["interest"] * inf +
                 w["difficulty"] * df +
                 w["workload"] * wf)

        return {
            "confidence": round(score, 3),
            "breakdown": {
                "grade_factor":      {"value": round(gf, 3),  "weight": w["grade"],      "reason": g_reason},
                "interest_factor":   {"value": round(inf, 3), "weight": w["interest"],   "reason": i_reason},
                "difficulty_factor": {"value": round(df, 3),  "weight": w["difficulty"], "reason": d_reason},
                "workload_factor":   {"value": round(wf, 3),  "weight": w["workload"],   "reason": w_reason},
            }
        }


# ─────────────────────────────────────────────
#  INFERENCE ENGINE
# ─────────────────────────────────────────────

@dataclass
class Recommendation:
    course: Course
    eligible: bool
    block_reasons: list[str]
    confidence: float
    confidence_breakdown: dict
    priority: str   # "highly recommended" | "recommended" | "optional" | "not eligible"

    def confidence_label(self) -> str:
        if self.confidence >= 0.75:
            return "🟢 High"
        elif self.confidence >= 0.50:
            return "🟡 Medium"
        elif self.confidence >= 0.30:
            return "🟠 Low"
        return "🔴 Very Low"


class InferenceEngine:
    """
    Combines rule engine (hard logic) + probabilistic reasoner (soft scoring)
    to generate and rank recommendations.
    """

    PRIORITY_THRESHOLDS = {
        "highly recommended": 0.75,
        "recommended":        0.50,
        "optional":           0.30,
    }

    def __init__(self):
        self.rule_engine       = RuleEngine(COURSE_DB)
        self.prob_reasoner     = ProbabilisticReasoner(COURSE_DB)

    def _assign_priority(self, eligible: bool, confidence: float) -> str:
        if not eligible:
            return "not eligible"
        for label, threshold in self.PRIORITY_THRESHOLDS.items():
            if confidence >= threshold:
                return label
        return "optional"

    def recommend(self, student: StudentProfile,
                  candidate_codes: list[str] = None,
                  top_n: int = 5) -> list[Recommendation]:
        """
        Generate recommendations for a student.
        If candidate_codes is None, evaluates all courses in the DB.
        """
        candidates = candidate_codes or list(COURSE_DB.keys())

        recommendations = []
        for code in candidates:
            if code not in COURSE_DB:
                continue
            course = COURSE_DB[code]

            eligible, block_reasons = self.rule_engine.evaluate(course, student)
            prob_result = self.prob_reasoner.compute(course, student)
            confidence  = prob_result["confidence"] if eligible else 0.0
            priority    = self._assign_priority(eligible, confidence)

            recommendations.append(Recommendation(
                course=course,
                eligible=eligible,
                block_reasons=block_reasons,
                confidence=confidence,
                confidence_breakdown=prob_result["breakdown"],
                priority=priority,
            ))

        # Sort: eligible first, then by confidence descending
        recommendations.sort(key=lambda r: (0 if r.eligible else 1, -r.confidence))
        return recommendations[:top_n] if top_n else recommendations


# ─────────────────────────────────────────────
#  USER INTERFACE  (Console)
# ─────────────────────────────────────────────

class SmartCourseRecommendationUI:

    BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       SMART COURSE RECOMMENDATION LOGIC                      ║
║       Rule-Based + Probabilistic Reasoning | Transparent AI  ║
╚══════════════════════════════════════════════════════════════╝
"""

    def __init__(self):
        self.engine = InferenceEngine()
        self.student: Optional[StudentProfile] = None

    # ─── Helpers ────────────────────────────────

    def _input(self, prompt: str) -> str:
        return input(f"  {prompt}").strip()

    def _header(self, title: str):
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    def _list_courses(self):
        self._header("Available Courses")
        for code, c in COURSE_DB.items():
            prereq_str = ", ".join(c.prerequisites) if c.prerequisites else "None"
            print(f"  {code:10} | {c.name:42} | Diff: {c.difficulty} | Prereqs: {prereq_str}")

    # ─── Student Setup ───────────────────────────

    def _setup_student(self):
        self._header("Student Profile Setup")

        # Student ID — must not be blank
        while True:
            sid = self._input("Student ID           : ")
            if sid:
                break
            print("  ⚠️  Student ID cannot be empty.")

        # Full Name — must not be blank
        while True:
            name = self._input("Full Name            : ")
            if name:
                break
            print("  ⚠️  Name cannot be empty.")

        # Completed courses — validate each code against DB
        print()
        print("  Enter completed course codes separated by commas (e.g. CS101,MATH101)")
        print("  Leave blank if none.")
        while True:
            raw_completed = self._input("Completed Courses    : ")
            parsed = [c.strip().upper() for c in raw_completed.split(",") if c.strip()]
            invalid = [c for c in parsed if c not in COURSE_DB]
            if invalid:
                print(f"  ⚠️  Unknown code(s): {', '.join(invalid)}. Use option 2 to see valid codes.")
            else:
                completed = parsed
                break

        # Grades — must be numeric and in range 0.0–10.0
        grades: dict[str, float] = {}
        if completed:
            print("\n  Enter grades for completed courses (0.0–10.0). Press Enter for default 7.0.")
            for code in completed:
                while True:
                    raw = self._input(f"  Grade for {code:10}: ")
                    if not raw:
                        grades[code] = 7.0
                        print(f"    → Default 7.0 applied for {code}")
                        break
                    try:
                        g = float(raw)
                        if not (0.0 <= g <= 10.0):
                            print(f"  ⚠️  Grade must be 0.0–10.0. Got {g}.")
                        else:
                            grades[code] = g
                            break
                    except ValueError:
                        print(f"  ⚠️  '{raw}' is not a valid number (e.g. 7.5).")

        # Semester — must be integer 1–8
        while True:
            raw_sem = self._input("\nCurrent Semester (1–8): ")
            try:
                semester = int(raw_sem)
                if 1 <= semester <= 8:
                    break
                print(f"  ⚠️  Semester must be between 1 and 8. Got {semester}.")
            except ValueError:
                print(f"  ⚠️  '{raw_sem}' is not a valid integer.")

        # Interests
        print("\n  Domains: programming | mathematics | ai | data_science | databases | systems")
        raw_interests = self._input("Interests (comma-sep) : ")
        interests = [i.strip().lower() for i in raw_interests.split(",") if i.strip()]
        if not interests:
            print("  → No interests entered; all domains treated equally.")

        # Workload — must be one of the three valid values
        print("\n  Workload preference: light | medium | heavy")
        while True:
            workload = self._input("Workload Preference  : ").lower()
            if workload in ("light", "medium", "heavy"):
                break
            print(f"  ⚠️  '{workload}' is invalid. Enter: light, medium, or heavy.")

        self.student = StudentProfile(
            student_id=sid,
            name=name,
            completed_courses=completed,
            grades=grades,
            current_semester=semester,
            interests=interests,
            workload_preference=workload,
        )
        print(f"\n  ✅ Profile created for {name}!")

    # ─── Recommendation Display ──────────────────

    def _show_recommendations(self, recs: list[Recommendation], verbose: bool = False):
        eligible   = [r for r in recs if r.eligible]
        ineligible = [r for r in recs if not r.eligible]

        self._header("📚 Recommended Courses")
        if not eligible:
            print("  No eligible courses found based on current profile.")
        for i, rec in enumerate(eligible, 1):
            print(f"\n  {i}. [{rec.priority.upper()}]  {rec.course.code} — {rec.course.name}")
            print(f"     Confidence: {rec.confidence_label()}  ({rec.confidence*100:.1f}%)")
            print(f"     Credits: {rec.course.credits}  |  Difficulty: {'★'*rec.course.difficulty}{'☆'*(5-rec.course.difficulty)}")
            print(f"     {rec.course.description}")
            if verbose:
                print("     ── Confidence Breakdown ──")
                for factor, data in rec.confidence_breakdown.items():
                    contrib = data["value"] * data["weight"]
                    print(f"       {factor:22} → {data['reason']}")
                    print(f"       {'':22}    contribution: {contrib*100:.1f}%")

        if ineligible:
            self._header("🚫 Ineligible Courses (with reasons)")
            for rec in ineligible:
                print(f"\n  ✗ {rec.course.code} — {rec.course.name}")
                for reason in rec.block_reasons:
                    print(f"      ↳ {reason}")

    # ─── Course Detail View ──────────────────────

    def _course_detail(self):
        code = self._input("Enter course code to inspect: ").upper().strip()
        if not code:
            print("  ⚠️  No code entered.")
            return
        if code not in COURSE_DB:
            print(f"  ⚠️  Course '{code}' not found. Use option 2 to see valid codes.")
            return
        c = COURSE_DB[code]
        self._header(f"Course Detail: {c.code}")
        print(f"  Name        : {c.name}")
        print(f"  Credits     : {c.credits}")
        print(f"  Difficulty  : {'★'*c.difficulty}{'☆'*(5-c.difficulty)} ({c.difficulty}/5)")
        print(f"  Domain      : {c.domain}")
        print(f"  Description : {c.description}")
        prereq_names = [f"{p} ({COURSE_DB[p].name})" for p in c.prerequisites if p in COURSE_DB]
        print(f"  Prerequisites: {', '.join(prereq_names) if prereq_names else 'None'}")

        if self.student:
            print()
            eligible, reasons = self.engine.rule_engine.evaluate(c, self.student)
            prob = self.engine.prob_reasoner.compute(c, self.student)
            status = "✅ Eligible" if eligible else "🚫 Not Eligible"
            print(f"  Eligibility for {self.student.name}: {status}")
            if reasons:
                for r in reasons:
                    print(f"    ↳ {r}")
            print(f"  Confidence Score: {prob['confidence']*100:.1f}%")

    # ─── Prerequisite Path Finder ────────────────

    def _prereq_path(self):
        code = self._input("Find prerequisite path to course: ").upper().strip()
        if not code:
            print("  ⚠️  No code entered.")
            return
        if code not in COURSE_DB:
            print(f"  ⚠️  Course '{code}' not found. Use option 2 to see valid codes.")
            return

        def get_path(code, visited=None):
            if visited is None:
                visited = set()
            if code not in COURSE_DB or code in visited:
                return []
            visited.add(code)
            c = COURSE_DB[code]
            path = []
            for prereq in c.prerequisites:
                path.extend(get_path(prereq, visited))
            path.append(code)
            return path

        path = get_path(code)
        self._header(f"Learning Pathway → {code}")
        for i, step in enumerate(path):
            c = COURSE_DB[step]
            tick = "✅" if self.student and step in self.student.completed_courses else "○"
            print(f"  {'  ' * i}{tick}  Step {i+1}: {step} — {c.name}  (Difficulty: {c.difficulty})")

    # ─── Export Results ──────────────────────────

    def _export_json(self, recs: list[Recommendation]):
        output = {
            "student": {
                "id":   self.student.student_id,
                "name": self.student.name,
                "completed_courses": self.student.completed_courses,
                "semester": self.student.current_semester,
            },
            "recommendations": [
                {
                    "code":       r.course.code,
                    "name":       r.course.name,
                    "eligible":   r.eligible,
                    "priority":   r.priority,
                    "confidence": r.confidence,
                    "block_reasons": r.block_reasons,
                }
                for r in recs
            ]
        }
        fname = f"recommendations_{self.student.student_id}.json"
        try:
            with open(fname, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\n  ✅ Results exported to {fname}")
        except PermissionError:
            print(f"  ⚠️  Permission denied writing '{fname}'. Try a different folder.")
        except OSError as e:
            print(f"  ⚠️  Could not save file: {e}")

    # ─── Main Menu ───────────────────────────────

    def run(self):
        print(self.BANNER)
        last_recs = []

        try:
            while True:
                print("\n  MAIN MENU")
                print("  1. Set up / update student profile")
                print("  2. List all courses")
                print("  3. Get course recommendations")
                print("  4. Get recommendations (verbose — show confidence breakdown)")
                print("  5. Inspect a specific course")
                print("  6. Show prerequisite learning path")
                print("  7. Export last recommendations to JSON")
                print("  8. Run demo with sample student")
                print("  0. Exit")

                choice = self._input("\nChoice: ")

                if choice == "0":
                    print("\n  Goodbye! 👋\n")
                    break

                elif choice == "1":
                    self._setup_student()

                elif choice == "2":
                    self._list_courses()

                elif choice in ("3", "4"):
                    if not self.student:
                        print("  ⚠️  Please set up a student profile first (option 1).")
                        continue
                    verbose = (choice == "4")
                    raw_n = self._input("How many top recommendations? [default 5]: ")
                    try:
                        n = int(raw_n)
                        if n <= 0:
                            print("  ⚠️  Must be a positive number. Using default 5.")
                            n = 5
                        elif n > len(COURSE_DB):
                            print(f"  ⚠️  Only {len(COURSE_DB)} courses exist. Showing all.")
                            n = len(COURSE_DB)
                    except ValueError:
                        if raw_n:
                            print(f"  ⚠️  '{raw_n}' is not a valid number. Using default 5.")
                        n = 5
                    last_recs = self.engine.recommend(self.student, top_n=n)
                    self._show_recommendations(last_recs, verbose=verbose)

                elif choice == "5":
                    self._course_detail()

                elif choice == "6":
                    self._prereq_path()

                elif choice == "7":
                    if not self.student or not last_recs:
                        print("  ⚠️  No recommendations generated yet.")
                        continue
                    self._export_json(last_recs)

                elif choice == "8":
                    self._run_demo()

                else:
                    print("  Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\n  Keyboard interrupt. Goodbye! 👋\n")

    # ─── Demo ────────────────────────────────────

    def _run_demo(self):
        self._header("DEMO — Sample Student: Aisha")
        self.student = StudentProfile(
            student_id="S2024001",
            name="Aisha Reddy",
            completed_courses=["CS101", "MATH101", "MATH102", "CS201"],
            grades={"CS101": 8.5, "MATH101": 9.0, "MATH102": 7.5, "CS201": 8.0},
            current_semester=4,
            interests=["ai", "programming"],
            workload_preference="medium",
        )
        print(f"""
  Student  : {self.student.name}
  Completed: {', '.join(self.student.completed_courses)}
  Interests: {', '.join(self.student.interests)}
  Workload : {self.student.workload_preference}
""")
        recs = self.engine.recommend(self.student, top_n=6)
        self._show_recommendations(recs, verbose=True)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = SmartCourseRecommendationUI()
    app.run()
