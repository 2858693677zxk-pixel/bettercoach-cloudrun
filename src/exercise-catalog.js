const EXERCISES = [
  { id: "machine_chest_press", name: "器械推胸", family: "press", bodyPart: "胸部", angle: "正前方或侧前方" },
  { id: "machine_crunch", name: "器械卷腹", family: "core_flexion", bodyPart: "腹部", angle: "正侧面" },
  { id: "standing_hip_abduction", name: "站姿髋外展", family: "isolation_hip", bodyPart: "臀部", angle: "斜前方" },
  { id: "seated_hip_abduction", name: "坐姿髋外展", family: "isolation_hip", bodyPart: "臀部", angle: "斜前方" },
  { id: "chest_supported_row", name: "胸托器械划船", family: "pull", bodyPart: "背部", angle: "侧前方" },
  { id: "t_bar_row", name: "支架杠划船", family: "pull", bodyPart: "背部", angle: "侧前方" },
  { id: "plate_loaded_pulldown", name: "悍马机正手下拉", family: "pull", bodyPart: "背部", angle: "侧前方" },
  { id: "plate_loaded_romanian_deadlift", name: "挂片式罗马尼亚硬拉", family: "hinge", bodyPart: "臀腿", angle: "正侧面" },
  { id: "single_arm_pulldown", name: "单臂下拉", family: "pull", bodyPart: "背部", angle: "正前方或斜前方" },
  { id: "hack_squat", name: "哈克深蹲", family: "squat", bodyPart: "腿部 / 臀部", angle: "侧面 30-45°" },
  { id: "hip_thrust", name: "臀桥", family: "hinge", bodyPart: "臀部", angle: "侧面 30-45°" },
  { id: "back_extension", name: "山羊挺身", family: "hinge", bodyPart: "下背部 / 臀腿后侧", angle: "侧面 30-45°" },
  { id: "plate_loaded_rear_leg_raise", name: "挂片后抬腿", family: "hinge", bodyPart: "臀腿", angle: "侧面 30-45°" },
  { id: "preacher_curl", name: "牧师凳二头弯举", family: "isolation_elbow", bodyPart: "手臂", angle: "侧面或侧前方" },
  { id: "single_arm_hammer_row", name: "单手悍马划船", family: "pull", bodyPart: "背部", angle: "侧前方或斜前方" },
  { id: "barbell_squat", name: "杠铃深蹲", family: "squat", bodyPart: "腿部", angle: "侧后方 30-45°" },
  { id: "goblet_squat", name: "高脚杯深蹲", family: "squat", bodyPart: "腿部", angle: "侧前方 30-45°" },
  { id: "deadlift", name: "硬拉", family: "hinge", bodyPart: "背部 / 腿部", angle: "侧面 30-45°" },
  { id: "romanian_deadlift", name: "罗马尼亚硬拉", family: "hinge", bodyPart: "腿部 / 臀部", angle: "侧面 30-45°" },
  { id: "bench_press", name: "杠铃卧推", family: "press", bodyPart: "胸部", angle: "侧前方 30-45°" },
  { id: "dumbbell_press", name: "哑铃卧推", family: "press", bodyPart: "胸部", angle: "侧前方" },
  { id: "shoulder_press", name: "肩上推举", family: "press", bodyPart: "肩部", angle: "侧前方" },
  { id: "push_up", name: "俯卧撑", family: "press", bodyPart: "胸部", angle: "侧前方" },
  { id: "dip", name: "双杠臂屈伸", family: "press", bodyPart: "胸部 / 肱三头肌", angle: "侧前方" },
  { id: "row", name: "划船", family: "pull", bodyPart: "背部", angle: "侧前方 30-45°" },
  { id: "open_elbow_row", name: "开肘划船", family: "pull", bodyPart: "上背部", angle: "斜前方" },
  { id: "lat_pulldown", name: "高位下拉", family: "pull", bodyPart: "背部", angle: "正前方或斜前方" },
  { id: "pull_up", name: "引体向上", family: "pull", bodyPart: "背部", angle: "正前方或斜前方" },
  { id: "face_pull", name: "面拉", family: "pull", bodyPart: "肩部 / 上背部", angle: "正前方" },
  { id: "lateral_raise", name: "侧平举", family: "isolation_shoulder", bodyPart: "肩部", angle: "正前方" },
  { id: "y_raise", name: "斜上举", family: "isolation_shoulder", bodyPart: "肩部", angle: "正前方或轻微斜前方" },
  { id: "fly", name: "飞鸟", family: "isolation_shoulder", bodyPart: "胸部 / 肩部", angle: "正前方或斜前方" },
  { id: "biceps_curl", name: "肱二头肌弯举", family: "isolation_elbow", bodyPart: "手臂", angle: "侧前方" },
  { id: "triceps_extension", name: "肱三头肌拉伸", family: "isolation_elbow", bodyPart: "手臂", angle: "侧前方" },
  { id: "leg_extension", name: "腿屈伸", family: "isolation_knee", bodyPart: "腿部", angle: "侧前方" },
  { id: "leg_curl", name: "腿弯举", family: "isolation_knee", bodyPart: "腿部", angle: "侧前方" },
  { id: "other", name: "自定义动作", family: "general", bodyPart: "全身", angle: "稳定拍摄完整工作组" }
];

const CAMERA_ANGLES = [
  { id: "side", name: "正侧面" },
  { id: "side_front", name: "侧前方 30-45°" },
  { id: "side_rear", name: "侧后方 30-45°" },
  { id: "front", name: "正前方" },
  { id: "front_oblique", name: "斜前方" },
  { id: "rear", name: "正后方" },
  { id: "unknown", name: "不确定" }
];

const EXERCISE_MAP = new Map(EXERCISES.map((item) => [item.id, item]));

function getExercise(id) {
  return EXERCISE_MAP.get(id) || EXERCISE_MAP.get("other");
}

module.exports = {
  CAMERA_ANGLES,
  EXERCISES,
  getExercise
};
