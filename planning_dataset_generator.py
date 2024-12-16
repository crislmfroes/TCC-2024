from datasets import load_dataset
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

dataset = load_dataset("BatsResearch/planetarium", split="train")

for i in range(len(dataset)):
    row = dataset[i]
    with open('/tmp/problem.pddl', 'w') as f:
        f.write(row['problem_pddl'])
    reader = PDDLReader()
    problem = reader.parse_problem(domain_filename=f'alfred.pddl', problem_filename='problem.pddl')
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)
        for action in result.plan.actions:
            print(help(action))
            exit()
        break