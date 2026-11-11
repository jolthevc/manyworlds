"""Turn raw research items into a compact, numbers-first summary for the Brain.

Engagement rate = (likes + comments + shares) / views when views exist (video),
otherwise / followers. Medians, not means: a single viral post shouldn't decide.
No handles or identifiable data leave this function; captions appear only as short,
anonymous style samples.
"""

from __future__ import annotations

from typing import Any


def summarize(conn, *, since_days: int = 30, sample_captions: int = 12) -> dict[str, Any]:
    by_format = conn.execute(
        """
        WITH er AS (
            SELECT platform, media_type, query_tag,
                   (coalesce(likes,0) + coalesce(comments,0) + coalesce(shares,0))::float
                     / NULLIF(coalesce(views, author_followers), 0) AS rate,
                   views, author_followers
            FROM research_items WHERE fetched_at > now() - make_interval(days => %s)
        )
        SELECT platform, media_type, count(*) AS posts,
               round((percentile_cont(0.5) WITHIN GROUP (ORDER BY rate))::numeric, 4) AS median_engagement,
               round((percentile_cont(0.5) WITHIN GROUP (ORDER BY views))::numeric) AS median_views
        FROM er WHERE rate IS NOT NULL
        GROUP BY platform, media_type ORDER BY platform, posts DESC
        """, (since_days,)).fetchall()

    by_tag = conn.execute(
        """
        SELECT platform, query_tag, count(*) AS posts,
               round((percentile_cont(0.5) WITHIN GROUP (ORDER BY
                   (coalesce(likes,0)+coalesce(comments,0)+coalesce(shares,0))::float
                   / NULLIF(coalesce(views, author_followers), 0)))::numeric, 4) AS median_engagement,
               round(avg(CASE WHEN media_type = 'video' THEN 1 ELSE 0 END)::numeric, 2) AS video_share
        FROM research_items WHERE fetched_at > now() - make_interval(days => %s)
        GROUP BY platform, query_tag ORDER BY median_engagement DESC NULLS LAST
        """, (since_days,)).fetchall()

    co_tags = conn.execute(
        """
        SELECT t AS hashtag, count(*) AS posts FROM research_items, unnest(hashtags) t
        WHERE fetched_at > now() - make_interval(days => %s)
        GROUP BY t ORDER BY posts DESC LIMIT 40
        """, (since_days,)).fetchall()

    samples = conn.execute(
        """
        SELECT platform, media_type, left(caption, 220) AS caption
        FROM research_items
        WHERE fetched_at > now() - make_interval(days => %s) AND length(coalesce(caption,'')) > 20
        ORDER BY (coalesce(likes,0)+coalesce(comments,0))::float / NULLIF(coalesce(views, author_followers), 0)
                 DESC NULLS LAST
        LIMIT %s
        """, (since_days, sample_captions)).fetchall()

    total = conn.execute("SELECT count(*) AS n FROM research_items WHERE fetched_at > now() - make_interval(days => %s)",
                         (since_days,)).fetchone()["n"]
    return {"total_posts": total, "by_format": by_format, "by_hashtag": by_tag,
            "top_co_hashtags": co_tags, "top_caption_samples": samples}
