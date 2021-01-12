#!/bin/python3

# Send a nicely formatted Slack message from the CHANGELOG.md file.
#
# Test
# ----
#
# ./slack-notification.py \
#     --webhook $SLACK_WEBHOOK_URL \
#     --infile ../../../CHANGELOG.md \
#     --channel '#test-github-notification'


import marko
import requests
import argparse
from htmlslacker import HTMLSlacker
from os import getenv


parser = argparse.ArgumentParser(description='Send Slack notification from CHANGELOG.md')

parser.add_argument('--infile', type=str, default="CHANGELOG.md",
                    help='The markdown file to read from.')

parser.add_argument('--channel', type=str, default="#general",
                    help='The Slack Channel to post to')

parser.add_argument('--name', type=str, default="AAW Updates",
                    help='The "User" of the message')

parser.add_argument('--webhook', type=str, default=getenv("SLACK_WEBHOOK_URL"),
                    help='The Slack Webhook URL to POST to.')

parser.add_argument('--url', type=str, default="<https://github.com/StatCan/daaas/blob/master/CHANGELOG.md|Changelog>",
                    help='A Slack-Readable link to the changelog.')

args = parser.parse_args()



# Stuff for the slack build kit format
section = lambda s: { "type" : "section", "text" : { "type" : "mrkdwn", "text" : s } }
divider = { "type" : "divider" }

CHANGELOG_URL = args.url
slack_builder = [
    section(f"*There are updates to the AAW!!! Check out the {CHANGELOG_URL} to see what's new!*")
]



# Get the *first* H1 Heading from the Changelog
with open(args.infile) as f:
    # Only use the first h1 header
    lines = f.readlines()
    last = len(lines)
    count = 0
    for (i, line) in enumerate(lines):
        if line.startswith("# "):
            count += 1
            if count == 2:
                last = i
                break
    # If there's only one h1, then this is idempotent
    lines = lines[:last]
    doc = marko.parse(''.join(lines)).children


# Parsing the markdown
render = lambda x: HTMLSlacker(marko.render(x)).get_output()

def flatten(x):
    if not isinstance(x, list):
        return [x]
    else:
        return sum( [ flatten(e.children) for e in x ] , [])

def list_render(l):
    blocks = [ flatten(x.children) for x in l ]
    print(blocks)
    for block in blocks:
        block[0] = '*' + block[0] + '*'
        for i in range(1, len(block)):
            block[i] = '• ' + block[i]

    return '\n\n'.join( '\n'.join(block) for block in blocks )

for e in doc:
    if isinstance(e, marko.block.Heading):
        slack_builder.append(divider)
        item = "*" + render(e) + "*"
    elif isinstance(e, marko.block.Quote):
        item = "> " + render(e)
    elif isinstance(e, marko.block.BlankLine):
        continue
    elif isinstance(e, marko.block.List):
        item = list_render(e.children)
    else:
        item = render(e)
    slack_builder.append(section(item))

print(slack_builder)


# Push!
payload = {
    "channel": args.channel,
    "username": args.name,
    "blocks": slack_builder,
    "icon_emoji": ":rocket:"
}

x = requests.post(args.webhook, json=payload)
print(x)
