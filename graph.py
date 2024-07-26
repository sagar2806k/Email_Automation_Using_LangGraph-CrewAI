from dotenv import load_dotenv
from langgraph.graph import StateGraph
from nodes import Nodes
from crew import EmailFilterCrew

from typing import List, Dict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict


class EmailsState(TypedDict):
    checked_emails_ids: List[str]
    emails: List[Dict]
    action_required_emails: Dict[str, Dict]

    def __init__(self):
        self.checked_emails_ids = []
        self.emails = []
        self.action_required_emails = {}

class AgentState(EmailsState):
    message: Annotated[Sequence[BaseMessage], operator.add]

class WorkFlow:
    def __init__(self):
    
        nodes = Nodes()
        workflow = StateGraph(AgentState)

        workflow.add_node("check_new_emails", nodes.check_email)
        workflow.add_node("wait_next_run", nodes.wait_next_run)
        workflow.add_node("draft_responses", EmailFilterCrew().kickoff)

        workflow.add_edge('draft_responses', 'wait_next_run')
        workflow.add_edge('wait_next_run', 'check_new_emails')

        workflow.set_entry_point("check_new_emails")
        workflow.add_conditional_edges(
            "check_new_emails",
            nodes.new_emails,
            {
                "continue": 'draft_responses',
                "end": "wait_next_run"
            }
        )

       #workflow.set_finish_point("draft_responses")
        self.app = workflow.compile()

if __name__ == "__main__":
    app = WorkFlow().app
