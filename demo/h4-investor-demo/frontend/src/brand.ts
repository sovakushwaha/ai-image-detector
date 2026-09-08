export const brand = {
  name: 'SOVA VERIFY',
  subtitle: 'AI Image Integrity Research',
  tagline: 'Know what you’re looking at.',
  heroSupport: 'AI-image forensic signals powered by a frozen research ensemble.',
  prototypeBadge: 'Research Prototype',
  authenticityWarning: 'A low AI score does not prove that an image is authentic.',
  scoreLabel: 'AI Detection Score',
} as const

export const researchMetrics = {
  developmentClean: {
    label: 'H4 development — CLEAN',
    dataset: 'V2 development evaluation folds',
    auc: 0.926,
    aiRecall: 0.525,
    realSpec: 0.975,
  },
  strongRobust: {
    label: 'H4 development — StrongRobust mean',
    dataset: 'V2 development transforms (JPEG / resize / blur / screenshot)',
    auc: 0.826,
    realSpec: 0.883,
  },
  ntire: {
    label: 'H4 sealed external — NTIRE',
    dataset: 'deepfakesMSU/NTIRE-RobustAIGenDetection-val (reserved revision)',
    auc: 0.625,
    aiRecall: 0.13,
    realSpec: 0.956,
    balAcc: 0.543,
  },
  comparabilityNote:
    'Development and NTIRE are different datasets and should not be interpreted as a paired comparison.',
} as const
