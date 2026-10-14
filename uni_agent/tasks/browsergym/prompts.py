"""Shared prompt templates for the BrowserGym task (row preprocessing + runtime).

The action list matches browsergym >= 0.4's high-level action space (the
``stop(answer)`` action of 0.3.x is gone: answers are sent with
``send_msg_to_user`` and the episode is ended with the ``submit`` tool).
"""

from __future__ import annotations

SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a web browser to solve tasks.
""".strip()

USER_PROMPT_TEMPLATE = """\
You are controlling a web browser through the `browser` tool to complete the task below.

Action space -- one action string per `browser` tool call. Element ids (`bid`) \
are the numbers in square brackets in the observation:

- click(bid='N')
- fill(bid='N', value='TEXT')               # clear a field, then type TEXT
- select_option(bid='N', options='X')       # pick from a <select>; a list also works: options=['X']
- hover(bid='N')
- press(bid='N', key_comb='Control+a')      # press keys on the focused element
- scroll(delta_x=0, delta_y=300)            # scroll the page by a pixel delta
- scroll_at(delta_x=0, delta_y=300, bid='N', coord_x=0, coord_y=0)  # scroll over one element
- new_tab() / tab_focus(index=0) / tab_close()
- goto(url='https://...') / go_back() / go_forward()
- send_msg_to_user(msg='ANSWER')            # send the task's answer to the user
- report_infeasible(reason='WHY')           # give up with a reason
- noop(wait_ms=1000)                        # wait (costs episode time)

Notes:
- The observation after each action is the page's accessibility tree. Rejected \
actions come back with an explanation -- adjust and retry.
- Some tasks are time-limited: act efficiently, do not loop on failed actions.
- When the task asks a question, answer with send_msg_to_user. When you believe \
the task is complete (or infeasible), call the `submit` tool to finish.

Task:

{goal}

Current page (url: {url}):

{obs}
"""


def build_messages(goal: str, obs: str, url: str = "") -> list[dict]:
    """Compose the episode-opening messages (goal + first page observation)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(goal=goal, obs=obs, url=url or "(none)")},
    ]
