const { EXERCISES } = require("./exercise-catalog");

// Each exercise owns an explicit prompt slot so action-specific coaching can be
// added incrementally without changing the shared review pipeline.
const ACTION_PROMPT_PROFILES = Object.freeze({
  machine_chest_press: { version: "placeholder-v1", instructions: [] },
  machine_crunch: { version: "placeholder-v1", instructions: [] },
  standing_hip_abduction: { version: "placeholder-v1", instructions: [] },
  seated_hip_abduction: { version: "placeholder-v1", instructions: [] },
  chest_supported_row: { version: "placeholder-v1", instructions: [] },
  t_bar_row: { version: "placeholder-v1", instructions: [] },
  plate_loaded_pulldown: { version: "placeholder-v1", instructions: [] },
  plate_loaded_romanian_deadlift: { version: "placeholder-v1", instructions: [] },
  single_arm_pulldown: { version: "placeholder-v1", instructions: [] },
  hack_squat: { version: "placeholder-v1", instructions: [] },
  hip_thrust: { version: "placeholder-v1", instructions: [] },
  back_extension: { version: "placeholder-v1", instructions: [] },
  plate_loaded_rear_leg_raise: { version: "placeholder-v1", instructions: [] },
  preacher_curl: { version: "placeholder-v1", instructions: [] },
  single_arm_hammer_row: { version: "placeholder-v1", instructions: [] },
  barbell_squat: { version: "placeholder-v1", instructions: [] },
  goblet_squat: { version: "placeholder-v1", instructions: [] },
  deadlift: { version: "placeholder-v1", instructions: [] },
  romanian_deadlift: { version: "placeholder-v1", instructions: [] },
  bench_press: { version: "placeholder-v1", instructions: [] },
  dumbbell_press: { version: "placeholder-v1", instructions: [] },
  shoulder_press: { version: "placeholder-v1", instructions: [] },
  push_up: { version: "placeholder-v1", instructions: [] },
  dip: { version: "placeholder-v1", instructions: [] },
  row: { version: "placeholder-v1", instructions: [] },
  open_elbow_row: { version: "placeholder-v1", instructions: [] },
  lat_pulldown: { version: "placeholder-v1", instructions: [] },
  pull_up: { version: "placeholder-v1", instructions: [] },
  face_pull: { version: "placeholder-v1", instructions: [] },
  lateral_raise: { version: "placeholder-v1", instructions: [] },
  y_raise: { version: "placeholder-v1", instructions: [] },
  fly: { version: "placeholder-v1", instructions: [] },
  biceps_curl: { version: "placeholder-v1", instructions: [] },
  triceps_extension: { version: "placeholder-v1", instructions: [] },
  leg_extension: { version: "placeholder-v1", instructions: [] },
  leg_curl: { version: "placeholder-v1", instructions: [] },
  other: { version: "placeholder-v1", instructions: [] }
});

function getActionPromptProfile(actionType) {
  return ACTION_PROMPT_PROFILES[actionType] || ACTION_PROMPT_PROFILES.other;
}

function validateActionPromptCoverage() {
  return EXERCISES
    .map((exercise) => exercise.id)
    .filter((actionType) => !Object.hasOwn(ACTION_PROMPT_PROFILES, actionType));
}

module.exports = {
  ACTION_PROMPT_PROFILES,
  getActionPromptProfile,
  validateActionPromptCoverage
};
