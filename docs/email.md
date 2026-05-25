# Email infrastructure

Design + setup notes for the welcome / unsubscribe / daily-digest stack.
Read this before touching `apps/api/src/email/` or running the
SES console steps in Stage 6.

## Goals

- Welcome new users with a one-shot transactional email on signup.
- Send an opt-out **daily digest** summarising what their brains did
  in the last 24 hours.
- Honour CAN-SPAM: one-click unsubscribe in every digest, per-user
  preference table, no marketing emails without explicit opt-in.

## Non-goals (for now)

- Multi-locale templates (English only at launch).
- In-app inbox / notification centre.
- Per-event push (a brain finished, a paper reproduction passed,
  etc.) — those collapse into the daily digest instead.
- Marketing campaigns. Transactional welcome + opt-out digest only.

## Provider: AWS SES

Picked over Resend / Postmark for cost at scale and AWS-native
ergonomics (we already use AWS for compute storage). Tradeoff is
heavier setup (DKIM / SPF / sandbox exit).

**Sender identity:** `noreply@profitops.ai`. We don't read replies
to that address — bounce / complaint handling routes through SES
SNS topics, not user replies.

## Data model

Two new tables, both additive (no destructive migration).

### `EmailPreference`

One row per (user, channel). When a user has no row for a channel,
default is **opted in** for transactional, **opted in** for digest
(EU users get a future opt-in flow — out of scope here).

```prisma
model EmailPreference {
  id              String     @id @default(cuid())
  userId          String
  channel         String     // "digest" | "welcome" | "system_alert"
  subscribed      Boolean    @default(true)
  unsubscribedAt  DateTime?
  unsubscribeReason String?  // free-form, e.g. "too frequent"
  updatedAt       DateTime   @updatedAt

  user            User       @relation(fields: [userId], references: [id], onDelete: Cascade)
  @@unique([userId, channel])
  @@index([userId])
}
```

### `EmailUnsubscribeToken`

One-time-use tokens for one-click unsubscribe links. We never
embed the user id in clear text — only a random opaque token.

```prisma
model EmailUnsubscribeToken {
  id           String     @id @default(cuid())
  token        String     @unique   // 32 random bytes, base64url
  userId       String
  channel      String                  // which preference this unsubscribes
  createdAt    DateTime   @default(now())
  expiresAt    DateTime                // 90 days from issue
  usedAt       DateTime?               // null = unused

  user         User       @relation(fields: [userId], references: [id], onDelete: Cascade)
  @@index([userId])
  @@index([expiresAt])
}
```

Tokens are minted per outbound email (so each digest carries a
fresh token). When a user clicks unsubscribe we:
1. Look up the token, verify it's not expired or used.
2. Mark `usedAt = now()`.
3. Set the matching `EmailPreference.subscribed = false`.
4. Render a confirmation page; offer a one-click re-subscribe in
   the same surface.

## Welcome email

**When fires:** `AuthService.register()`, after the `User` row commits
successfully. Wrapped in try/catch — a failed send must not roll back
the signup, only log a warning.

**Content (rough):**
- Subject: `Welcome to PFN Studio · build your first brain in 5 mins`
- Body: who we are (one sentence), link to `/teach` to build a first
  brain, link to `/docs/getting-started`, unsubscribe link for the
  digest channel (welcome itself is transactional + non-opt-out).

**Suppression:** if `EmailPreference{channel: 'welcome', subscribed: false}`
exists for the user (extremely rare — they'd have to set it before
signup, which isn't possible today), skip. Future-proofs the path.

## Daily digest

**Cadence:** one cron tick at **14:00 UTC** (≈ 8am Pacific). Picked
US-business-morning so the email lands at the start of the workday
for our launch demographic. Per-user timezone scheduling is a future
upgrade.

**Cron framework:** `@nestjs/schedule` with `@Cron(CronExpression.EVERY_DAY_AT_2PM, { timeZone: 'UTC' })`.

**Audience query:**
```
SELECT u.* FROM User u
LEFT JOIN EmailPreference p ON p.userId = u.id AND p.channel = 'digest'
WHERE p.subscribed IS DISTINCT FROM false                -- opted in (or no row)
  AND EXISTS (
    SELECT 1 FROM Run r
    JOIN Project pr ON r.projectId = pr.id
    JOIN OrgMember m ON m.orgId = pr.orgId AND m.userId = u.id
    WHERE r.createdAt > NOW() - INTERVAL '24 hours'
  )
```

A user with **no activity in the last 24h gets no email.** Avoids
the "empty digest" anti-pattern.

**Content:**
- Subject: `Your PFN Studio digest · N brains, M runs today`
- Body sections (only render if non-empty):
  - **Brains finished practising** — name + capability + grade letter
  - **Brains that matched the paper** — name + paper citation
  - **Brains that fell short** — name + the gap (e.g. "0.45 RMSE vs paper's 0.30")
  - **Pending runs** — anything still running > 6 hours (oncall hint)
- One-click unsubscribe in footer.

**Throttling:**
- SES sandbox is 1 email/sec / 200/day. **Cannot ship digest before
  production access.**
- Production access (default) gives 14 emails/sec, 50k/day. Loop
  with `p-throttle` at 10/sec to leave headroom for welcome emails.
- Add a `EmailSendLog` row per send for idempotency — never send
  the same user the same digest day twice.

```prisma
model EmailSendLog {
  id          String   @id @default(cuid())
  userId      String
  channel     String
  digestDate  DateTime?    // only set for digest sends
  sentAt      DateTime @default(now())
  sesMessageId String?     // for bounce correlation
  user        User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  @@unique([userId, channel, digestDate])
  @@index([userId])
}
```

The `@@unique([userId, channel, digestDate])` constraint guarantees
that if the cron job retries (process restart, etc.) we don't
double-send.

## Unsubscribe flow

1. Email body links to `https://app.profitops.ai/unsubscribe?token=<base64url>`.
2. Frontend `/unsubscribe` page calls `POST /api/email/unsubscribe`
   with the token in the body (avoids logging the token in CDN access
   logs).
3. API:
   - Validate token (exists, not expired, not used).
   - Mark token `usedAt`.
   - Update preference: `subscribed = false`, `unsubscribedAt = now()`.
   - Return `{ channel, email }` so the frontend can render
     "You've been unsubscribed from `digest` emails."
4. Confirmation page shows a "Re-subscribe" button that POSTs back
   to the same endpoint (with a different action verb) — anyone
   clicking unsubscribe by accident gets a one-click recovery.

**Token security:**
- 32 bytes from `crypto.randomBytes`, base64url-encoded → 43 chars.
- Indexed unique on the token column. Lookup is a single PK-ish hit.
- Single-use (`usedAt` flag) — re-clicking the same link shows
  "this link has already been used; current state: unsubscribed."
- 90-day expiry. After that the user has to wait for the next
  digest send for a fresh token.

## AWS SES setup checklist (Stage 6)

These are console / DNS steps. The code is gated behind env vars so
the API ships and runs locally without any of this. Production
sending needs all five.

- [ ] **Verify sending domain.** SES Console → Verified identities
      → Create identity → Domain → `profitops.ai`. Copy the 3 CNAME
      records SES gives you.
- [ ] **DNS — DKIM.** Add the 3 CNAME records to the Route53 (or
      whatever) zone. Wait for SES to mark the domain "Verified"
      (usually < 15 min, sometimes hours).
- [ ] **DNS — SPF.** Add a TXT record at `profitops.ai`:
      `v=spf1 include:amazonses.com -all`. If there's an existing SPF
      record, merge — there must be exactly one.
- [ ] **Request production access.** SES Console → Account
      dashboard → Request production access. Fill in the form
      truthfully (transactional welcome + opt-out daily digest;
      bounce + complaint webhooks wired). Approval is usually
      same-day.
- [ ] **Bounce + complaint webhook.** Create an SNS topic
      (`ses-bounces-prod`), subscribe a `POST /api/email/ses-webhook`
      endpoint to it. In SES → Configuration sets → Event
      destinations → SNS, route Bounce + Complaint events to it.
      Webhook handler verifies SNS signature, then sets the
      offending user's `EmailPreference.subscribed = false` so we
      stop sending immediately.
- [ ] **Env vars on the API host:**
      - `AWS_REGION=us-east-1` (or your SES region)
      - `AWS_ACCESS_KEY_ID=...`
      - `AWS_SECRET_ACCESS_KEY=...` (IAM user with `ses:SendEmail`,
        `ses:SendRawEmail` scoped to the verified identity)
      - `EMAIL_FROM=noreply@profitops.ai`
      - `EMAIL_REPLY_TO=hello@profitops.ai`  (optional)
      - `APP_BASE_URL=https://app.profitops.ai`  (used for unsubscribe
        + brain page links in digest content)

## Implementation stages (code)

1. **Schema + backup** — adds the three tables + indexes. Run
   `node scripts/db-backup.js` first per CLAUDE.md rules.
2. **`email/` module skeleton** — `SesService`, `EmailTemplateService`,
   env-var wiring. No senders yet. Module logs a warning at
   startup if SES env vars are missing.
3. **Welcome email** — `WelcomeEmailService.sendForUser(userId)`,
   called from `AuthService.register()` after the txn commits.
   Best-effort.
4. **Unsubscribe** — API endpoint + frontend page + token mint /
   verify helpers.
5. **Daily digest** — `DigestEmailService` + cron decorator +
   audience query + idempotency log + throttle.
6. **SES infra** — checklist above. Manual.

## Error handling

- **SES not configured** (env vars missing): module loads, logs a
  warning at startup, every `send*` call returns
  `{ ok: false, reason: 'ses_not_configured' }`. Welcome / digest
  callers log + carry on.
- **SES SendEmail throws** (rate limit, throttle, permission error):
  caught, logged with the AWS error code, no retry on the welcome
  path. Digest path retries once with backoff then drops the user
  from that day's send.
- **Bounce / complaint received** (via SNS webhook): user's
  preferences flipped to unsubscribed immediately. We do *not*
  delete the user.

## Observability

- Per-send log line: `email.sent channel=digest userId=u_123 sesMessageId=...`.
- Daily digest summary: `email.digest.summary tried=N sent=M skipped=K errored=E`.
- Bounce / complaint counter — surface to Grafana when oncall
  dashboards land.
