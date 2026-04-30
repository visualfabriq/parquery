#!/usr/bin/env bash
branch=$(git rev-parse --abbrev-ref HEAD)

if [[ "$branch" =~ ^(main|master|release/) ]]; then
    exit 0
fi

if [[ "$branch" =~ [A-Z]{2,}-[0-9]+ ]]; then
    exit 0
fi

echo "ERROR: Branch \"$branch\" must contain a JIRA ticket ID (e.g. FOUN-123-description)"
exit 1
