# Data

Everything here is hand-written for this repo and has the same MIT license as the code. The examples download nothing.

## `tickets.jsonl`: 108 support tickets for a fictional product, Tasklane

Six labels, 18 tickets each. The examples split them 6 / 6 / 6 per label into train, dev and test (36 each).

The labels follow a house policy that a model cannot know from the label names alone:

1. When a ticket touches several topics, the first matching rule wins:
   `security` > `billing` > `account` > `bug` > `feature` > `howto`.
2. `howto` is for things Tasklane can do. `feature` is for things it cannot.
   It has: boards, custom-interval recurring tasks, CSV export, Slack, SAML SSO (Business plan),
   guests, templates, authenticator-app 2FA, API tokens, archive restore (60 days), an audit log (Business).
   It does not have: Gantt or timeline views, time tracking, custom fields, dark mode, offline mode,
   Jira sync, PDF export, card dependencies, multiple assignees, a calendar view, email-to-card.

Half the tickets are plain. The other half mix topics or ask about a feature that looks missing but
exists, which is where an untuned prompt goes wrong and an optimizer has something to find.

## `help_docs.jsonl` and `help_qa.jsonl`: Tasklane help center

15 short documents and 36 questions (12 need arithmetic or combining facts; those rows have `derived: true`) about them, used by the RAG examples. Each question has a
reference `answer`, a short `key` fact (for the cheap containment metric), and the `doc_id` it comes from.
