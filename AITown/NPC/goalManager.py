from typing import List, Dict
#from NPC.goalNode import GoalNode

class GoalManager:
    def __init__(self, root_goal: str, decompose_fn):
        self.root = root_goal
        self.stack = [self.root]
        self.decompose_fn = decompose_fn
        self.current_reason = ""
        
    def get_current_goal(self):
        return self.stack[-1] if self.stack else None

    def get_parent_goal(self):
        if len(self.stack) > 1:
            return self.stack[-2]
        return None

    def get_root_goal(self):
        return self.root
    
    def push_goal(self, goal: str):
        self.stack.append(goal)
    
    def pop_goal(self):
        if self.stack:
            return self.stack.pop()