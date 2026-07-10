import os
import sys

errors = []

codeowners = ".github/CODEOWNERS"
if not os.path.isfile(codeowners):
    errors.append(
        f"COMMIT BLOCKED - {codeowners} is missing. Add at least one owner: * @your-github-username"
    )
else:
    with open(codeowners) as f:
        owners = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not owners:
        errors.append(
            f"COMMIT BLOCKED - {codeowners} has no owners defined. Add at least one: * @your-github-username"
        )

pr_template = ".github/pull_request_template.md"
if not os.path.isfile(pr_template):
    errors.append(f"COMMIT BLOCKED - {pr_template} is missing.")

for e in errors:
    print(e)

sys.exit(1 if errors else 0)
