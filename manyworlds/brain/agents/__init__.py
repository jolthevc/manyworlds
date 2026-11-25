"""Brain agents. Importing this package registers every agent (manyworlds.brain.agent.REGISTRY)."""

from . import analyst, breadth_driver, comment_themes, creator_lab, echo, evaluator, idea_reviser, intelligence, market_lead, music_critic, scouts, showrunner, strategist, style_director, talent_manager  # noqa: F401
# The music department (spec 2026-11-23-music-department.md, section 3): the song roles register themselves with
# brain.music_roles.SONG_ROLES too (the pipeline composes them).
from . import a_and_r, curator, head_of_music, lyricist_role, melody, mix_master, music_pipeline_agent, music_talent_manager, producer, songwriter, vocal_director  # noqa: F401
