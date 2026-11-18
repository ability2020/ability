const OWNER = "ability2020";
const REPO = "ability";
const WORKFLOW = "bitget-test.yml";
const BRANCH = "main";
const MAX_AGE_MINUTES = 25;

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

export default {
  async fetch(request, env) {
    if (request.method !== "GET") {
      return json({ ok: false, error: "Method not allowed" }, 405);
    }

    const supplied = request.headers.get("authorization") || "";
    if (!env.WATCHDOG_SECRET || supplied !== `Bearer ${env.WATCHDOG_SECRET}`) {
      return json({ ok: false, error: "Unauthorized" }, 401);
    }
    if (!env.GITHUB_TOKEN) {
      return json({ ok: false, error: "GITHUB_TOKEN secret is not configured" }, 500);
    }

    const headers = {
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "ability-paper-trading-watchdog",
    };

    try {
      const runsUrl = `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/runs?branch=${BRANCH}&per_page=1`;
      const runsResponse = await fetch(runsUrl, { headers });
      if (!runsResponse.ok) {
        return json({ ok: false, error: "Could not inspect latest workflow run", github_status: runsResponse.status }, 502);
      }

      const payload = await runsResponse.json();
      const latest = payload.workflow_runs?.[0] || null;
      const now = Date.now();

      if (latest) {
        const timestamp = latest.run_started_at || latest.created_at;
        const started = Date.parse(timestamp || "");
        if (!Number.isFinite(started)) {
          return json({ ok: false, error: "Latest workflow run has no usable timestamp" }, 502);
        }

        const ageMinutes = (now - started) / 60000;
        if (ageMinutes <= MAX_AGE_MINUTES) {
          return json({
            ok: true,
            action: "skipped_recent_run",
            latest_run_number: latest.run_number,
            latest_event: latest.event,
            latest_status: latest.status,
            age_minutes: Number(ageMinutes.toFixed(1)),
            threshold_minutes: MAX_AGE_MINUTES,
          });
        }
      }

      const dispatchUrl = `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`;
      const dispatchResponse = await fetch(dispatchUrl, {
        method: "POST",
        headers: {
          ...headers,
          "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ref: BRANCH,
          inputs: {
            run_historical_research: "false",
            trigger_source: "external_watchdog",
          },
        }),
      });

      if (dispatchResponse.status !== 204) {
        const detail = (await dispatchResponse.text()).slice(0, 500);
        return json({
          ok: false,
          error: "GitHub rejected the backup workflow dispatch",
          github_status: dispatchResponse.status,
          detail,
        }, 502);
      }

      return json({
        ok: true,
        action: "backup_dispatched",
        reason: latest ? "latest workflow run exceeded the freshness threshold" : "no previous workflow run found",
        latest_run_number: latest?.run_number ?? null,
        threshold_minutes: MAX_AGE_MINUTES,
        historical_research: false,
      });
    } catch (error) {
      return json({ ok: false, error: "Watchdog request failed", detail: String(error).slice(0, 300) }, 500);
    }
  },
};
