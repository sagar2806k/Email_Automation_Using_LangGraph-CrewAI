from crewai import Task
from textwrap import dedent

class EmailFilterTasks:
    def filter_emails_task(self, agent, emails):
        return Task(
            description=dedent(f"""\
                Analyze a batch of emails and filter out
                non-essential ones such as newsletters, promotional content, and notifications.

                Use your expertise in email content analysis to distinguish
                important emails from the rest, paying attention to the sender and avoiding invalid emails.

                Make sure to filter for the messages actually directed at the user and avoid notifications.

                EMAILS
                -------
                {emails}

                Your final answer MUST be the relevant thread_ids and the sender, using bullet points.
            """),
            agent=agent,
            expected_output=dedent("""\
                Type: list
                Description: A list of relevant thread_ids and their senders.
            """)
        )

    def action_required_emails_task(self, agent):
        return Task(
            description=dedent("""\
                For each email thread, pull and analyze the complete threads using only the actual Thread ID.
                Understand the context, key points, and the overall sentiment
                of the conversation.

                Identify the main query or concerns that need to be
                addressed in the response for each.

                Your final answer MUST be a list for all emails with:
                - the thread_id
                - a summary of the email thread
                - a highlighting of the main points
                - identify the user and who he will be answering to
                - communication style in the thread
                - the sender's email address
            """),
            agent=agent,
            expected_output=dedent("""\
                Type: list
                Description: A list of email threads with their thread_id, summary, main points, user and recipient, communication style, and sender's email address.
            """)
        )

    def draft_responses_task(self, agent):
        return Task(
            description=dedent("""\
                Based on the action-required emails identified, draft responses for each.
                Ensure that each response is tailored to address the specific needs
                and context outlined in the email.

                - Assume the persona of the user and mimic the communication style in the thread.
                - Feel free to do research on the topic to provide a more detailed response, IF NECESSARY.
                - IF research is necessary, do it BEFORE drafting the response.
                - If you need to pull the thread again, do it using only the actual Thread ID.

                Use the tool provided to draft each of the responses.
                When using the tool, pass the following input:
                - to (sender to be responded)
                - subject
                - message

                You MUST create all drafts before sending your final answer.
                Your final answer MUST be a confirmation that all responses have been drafted.
            """),
            agent=agent,
            expected_output=dedent("""\
                Type: confirmation
                Description: A confirmation that all responses have been drafted.
            """)
        )