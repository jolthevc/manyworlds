# Instagram setup (owner, once; then a few minutes per creator)

Publishing, metrics and comment replies use Meta's **Instagram API with Unified Login** (graph.facebook.com).
One system-user token covers every creator account in our business portfolio, so connecting a new creator never
needs a new token. Standard Access is enough because every account is ours: no App Review and no Business
Verification. (Meta docs, read 2026-11-21: instagram-platform overview, Unified Login authentication and permissions,
content publishing, comment moderation.)

## Once

1. **Business portfolio**: create one at business.facebook.com if you have none.
2. **Developer app**: at developers.facebook.com create a Business app owned by that portfolio and add the
   Instagram product, set up "Instagram API with Unified Login".
3. **System user**: Business Settings > Users > System users > Add (role Admin). Assign it the app.
4. **Token**: on the system user, choose Generate token for the app with these permissions:
   `instagram_basic`, `instagram_content_publish`, `instagram_manage_comments`, `instagram_manage_insights`
   (add `business_management` only if Meta asks for it). Pick a non-expiring token if the configuration offers
   one; otherwise note the expiry date.
5. **Store it** (never paste it in a terminal or chat): put it in `.env` as `META_SYSTEM_USER_TOKEN=...`, then
   pipe it to Fly: `grep '^META_SYSTEM_USER_TOKEN=' .env | ~/.fly/bin/flyctl.exe secrets import -a manyworlds-mw`.

## Per creator

1. `fly ssh console -a manyworlds-mw -C "mw account sheet <creator>"` prints the setup sheet: username, display
   name, bio (with the AI disclosure, within 150 characters), a link to the profile photo, and the checklist.
2. A person creates the account by hand from the sheet: a real phone or browser, no automation. They switch it to a
   professional (Creator) account and turn on Instagram's AI label on the profile if offered.
3. Business Settings > Accounts > Instagram accounts: add the account to the portfolio, then assign it to the
   system user.
4. `mw account connect <creator> --username <username>` checks everything through the API. It refuses to connect
   if the account isn't assigned to the app, the bio lacks the disclosure, or the token can't publish. It warns if
   there's no profile photo or the display name doesn't match the sheet.
5. Once connected, the creator leaves the sandbox. Posts go out within the ramp (policy `instagram.ramp`: 1 a day
   for the first week, 2 a day until day 21, then the planned cadence). Nothing is published while
   `operations.mode` is `paused`.

## Comments

`autonomy.comment_replies` is `draft`: the creator writes replies but they are stored, not sent. Read them with
`mw comments <creator>`. Set it to `autonomous` when they read right, or `disabled` to stop reading comments.
Replies never mention anyone, never link, never answer our own accounts, and say plainly that the creator is
AI-generated when asked.
