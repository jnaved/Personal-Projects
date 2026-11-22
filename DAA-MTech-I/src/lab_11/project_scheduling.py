"""
========================================================================================
DAA Lab 11 - Assignment 4: Project Scheduling Using a Directed Acyclic Graph (DAG)
========================================================================================

Problem Statement:
------------------
A software development team has tasks that depend on the completion of other tasks.
1. Represent the tasks and dependencies as a Directed Acyclic Graph (DAG).
2. Determine a valid task execution order (Topological Sort).
3. Calculate the earliest completion time of the project using the Critical Path Method (CPM).
4. Identify critical tasks (zero-slack bottleneck activities) and the critical path.

Graph Representation:
---------------------
- Vertices represent project tasks, each having an estimated duration (e.g., in days).
- Directed edges (u -> v) represent precedence constraints: task u must finish before
  task v can begin.

Algorithm (Topological Sort + Critical Path Method):
----------------------------------------------------
1. Cycle Detection & Topological Order:
   - Use Kahn's algorithm (in-degree based queue).
   - If not all tasks can be processed, a circular dependency (cycle) is detected.
2. Forward Pass (Earliest Times):
   - For tasks in topological order:
     * Earliest Start:  ES[v] = max({EF[u] for u in predecessors(v)}, default=0)
     * Earliest Finish: EF[v] = ES[v] + duration[v]
   - Earliest Project Completion Time: T_project = max({EF[v] for all v})
3. Backward Pass (Latest Times & Slack):
   - For tasks in reverse topological order:
     * Latest Finish: LF[v] = min({LS[w] for w in successors(v)}, default=T_project)
     * Latest Start:  LS[v] = LF[v] - duration[v]
     * Slack (Float): Slack[v] = LS[v] - ES[v] = LF[v] - EF[v]
4. Critical Path:
   - Tasks with Slack == 0 cannot be delayed without delaying the entire project.
========================================================================================
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Task:
    task_id: str
    name: str
    duration: float


class ProjectDAG:
    """Manages project tasks, precedence dependencies, and scheduling calculations."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.adj: Dict[str, List[str]] = defaultdict(list)  # u -> list of successors
        self.pred: Dict[str, List[str]] = defaultdict(list)  # v -> list of predecessors
        self.in_degree: Dict[str, int] = defaultdict(int)

    def add_task(self, task_id: str, name: str, duration: float) -> None:
        """Register a new task in the project."""
        if duration < 0:
            raise ValueError("Task duration cannot be negative.")
        self.tasks[task_id] = Task(task_id, name, duration)
        # Ensure presence in degree tracking
        if task_id not in self.in_degree:
            self.in_degree[task_id] = 0

    def add_dependency(self, prereq_id: str, dependent_id: str) -> None:
        """Add dependency: prereq_id must finish before dependent_id starts."""
        if prereq_id not in self.tasks or dependent_id not in self.tasks:
            raise KeyError("Both tasks must be added before declaring a dependency.")
        self.adj[prereq_id].append(dependent_id)
        self.pred[dependent_id].append(prereq_id)
        self.in_degree[dependent_id] += 1

    def topological_sort(self) -> Optional[List[str]]:
        """
        Finds a valid linear task execution order using Kahn's algorithm.
        Returns None if the graph contains a cycle (circular dependency).
        """
        in_deg = dict(self.in_degree)
        queue = deque([t for t in self.tasks if in_deg[t] == 0])
        topo_order: List[str] = []

        while queue:
            u = queue.popleft()
            topo_order.append(u)

            for v in self.adj[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)

        if len(topo_order) != len(self.tasks):
            return None  # Cycle detected

        return topo_order

    def calculate_schedule(self) -> Optional[Dict[str, Dict]]:
        """
        Performs Forward and Backward passes of the Critical Path Method (CPM).
        Returns schedule dictionary containing ES, EF, LS, LF, Slack for each task.
        """
        topo_order = self.topological_sort()
        if topo_order is None:
            return None

        # 1. Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
        es: Dict[str, float] = {}
        ef: Dict[str, float] = {}

        for u in topo_order:
            if not self.pred[u]:
                es[u] = 0.0
            else:
                es[u] = max(ef[p] for p in self.pred[u])
            ef[u] = es[u] + self.tasks[u].duration

        earliest_project_finish = max(ef.values()) if ef else 0.0

        # 2. Backward Pass: Latest Finish (LF) and Latest Start (LS)
        ls: Dict[str, float] = {}
        lf: Dict[str, float] = {}

        for u in reversed(topo_order):
            if not self.adj[u]:
                lf[u] = earliest_project_finish
            else:
                lf[u] = min(ls[s] for s in self.adj[u])
            ls[u] = lf[u] - self.tasks[u].duration

        # 3. Assemble schedule results
        schedule = {}
        for u in topo_order:
            slack = ls[u] - es[u]
            is_critical = abs(slack) < 1e-9
            schedule[u] = {
                "task": self.tasks[u],
                "ES": es[u],
                "EF": ef[u],
                "LS": ls[u],
                "LF": lf[u],
                "slack": slack,
                "is_critical": is_critical,
            }

        return schedule

    def find_critical_paths(self) -> List[List[str]]:
        """Finds all critical paths (paths composed entirely of critical tasks)."""
        schedule = self.calculate_schedule()
        if schedule is None:
            return []

        critical_tasks = {t for t, data in schedule.items() if data["is_critical"]}
        start_tasks = [t for t in critical_tasks if not self.pred[t]]
        end_tasks = {t for t in critical_tasks if not [s for s in self.adj[t] if s in critical_tasks]}

        paths: List[List[str]] = []

        def _dfs(curr: str, path: List[str]):
            if curr in end_tasks:
                paths.append(list(path))
                return
            for nxt in self.adj[curr]:
                if nxt in critical_tasks and schedule[nxt]["ES"] == schedule[curr]["EF"]:
                    path.append(nxt)
                    _dfs(nxt, path)
                    path.pop()

        for st in start_tasks:
            _dfs(st, [st])

        return paths


# ========================================================================================
# Demonstration / Driver Code
# ========================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ASSIGNMENT 4: PROJECT SCHEDULING USING DIRECTED ACYCLIC GRAPH (DAG)")
    print("=" * 80)

    project = ProjectDAG()

    # Define tasks for a Software Development Project (duration in days)
    tasks_data = [
        ("T1", "Requirements & Planning", 4.0),
        ("T2", "System Architecture Design", 5.0),
        ("T3", "Database Schema & Setup", 3.0),
        ("T4", "Backend API Development", 8.0),
        ("T5", "Frontend UI Design & Coding", 6.0),
        ("T6", "Integration & Testing", 5.0),
        ("T7", "User Acceptance Testing (UAT)", 3.0),
        ("T8", "Deployment & Release", 2.0),
    ]

    for tid, name, dur in tasks_data:
        project.add_task(tid, name, dur)

    # Define precedence dependencies (u -> v: u must finish before v starts)
    dependencies = [
        ("T1", "T2"),  # Architecture starts after Requirements
        ("T2", "T3"),  # Database follows Architecture
        ("T2", "T4"),  # Backend follows Architecture
        ("T2", "T5"),  # Frontend follows Architecture
        ("T3", "T4"),  # Backend needs Database setup
        ("T4", "T6"),  # Integration needs Backend API
        ("T5", "T6"),  # Integration needs Frontend UI
        ("T6", "T7"),  # UAT needs Integration & Testing
        ("T7", "T8"),  # Deployment follows UAT
    ]

    for prereq, dep in dependencies:
        project.add_dependency(prereq, dep)

    print("\nProject Tasks and Estimated Durations:")
    print("-" * 65)
    for tid, name, dur in tasks_data:
        preds = project.pred[tid]
        pred_str = ", ".join(preds) if preds else "None"
        print(f"  {tid:<3} : {name:<30} | {dur:>2.0f} days | Depends on: {pred_str}")

    # 1. Topological Sort (Valid Execution Order)
    topo_order = project.topological_sort()
    if topo_order is None:
        print("\n[ERROR] Circular dependency detected in the project graph!")
    else:
        print("\n" + "=" * 65)
        print("VALID TASK EXECUTION ORDER (Topological Sort):")
        print("=" * 65)
        print(" -> ".join(topo_order))
        print("Any task executes strictly after its prerequisites are met.")

    # 2. Critical Path Method Schedule Calculation
    schedule = project.calculate_schedule()
    if schedule:
        earliest_completion = max(data["EF"] for data in schedule.values())

        print("\n" + "=" * 80)
        print("PROJECT SCHEDULE & CRITICAL PATH ANALYSIS (CPM):")
        print("=" * 80)
        print(f"{'Task':<4} | {'Name':<28} | {'Dur':<4} | {'ES':<5} | {'EF':<5} | {'LS':<5} | {'LF':<5} | {'Slack':<5} | {'Status'}")
        print("-" * 80)
        for tid in topo_order:
            d = schedule[tid]
            status = "CRITICAL *" if d["is_critical"] else "Flexible"
            print(
                f"{tid:<4} | {d['task'].name:<28} | {d['task'].duration:>3.0f}d | "
                f"{d['ES']:>4.0f}d | {d['EF']:>4.0f}d | {d['LS']:>4.0f}d | {d['LF']:>4.0f}d | "
                f"{d['slack']:>4.0f}d | {status}"
            )
        print("-" * 80)
        print(f"EARLIEST PROJECT COMPLETION TIME: {earliest_completion:.0f} days")

        critical_paths = project.find_critical_paths()
        print("\nCritical Path(s) [Bottlenecks determining project duration]:")
        for cp in critical_paths:
            path_names = [f"{tid} ({project.tasks[tid].name})" for tid in cp]
            print(f"  * {' -> '.join(cp)}")
            print(f"    {' -> '.join(path_names)}")
    print("=" * 80)
