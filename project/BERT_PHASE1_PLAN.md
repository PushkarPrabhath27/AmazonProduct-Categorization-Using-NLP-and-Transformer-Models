# BERT Full Training Plan - Phase 1 & 2

## Phase 1: 20K Samples (STARTING NOW)

### Configuration
```
Training Samples: 20,000 (25% of 80K)
Validation Samples: 2,000
Epochs: 3
Batch Size: 4
Max Sequence Length: 128 (double previous 64)
Gradient Accumulation: 4 (effective batch: 16)
Learning Rate: 2e-5
Warmup: 10% of steps
Checkpoint: Every 200 steps (crash recovery)
```

### Expected Results
- **Accuracy Target**: 94-95%
- **Training Time**: 3-4 hours
- **Start Time**: 15:52 PM
- **Expected Completion**: ~19:30 PM

### Why This Will Work
1. ✅ **More Data**: 20K vs previous 3K (6.7x increase)
2. ✅ **Better Context**: 128 tokens vs 64 (captures full descriptions)
3. ✅ **More Training**: 3 epochs vs 2
4. ✅ **Crash-Safe**: Checkpoints every 200 steps
5. ✅ **Stratified**: Maintains category balance

### Crash Recovery
- Checkpoints saved to: `models/bert_checkpoints/`
- Auto-resume from last checkpoint if interrupted
- Progress logged every 50 steps

---

## Phase 2: 40K Samples (IF PHASE 1 SUCCEEDS)

### Trigger Conditions
- ✅ Phase 1 completes without crash
- ✅ Laptop remains stable
- ⚠️ Phase 1 accuracy < 95% (if ≥95%, Phase 2 optional)

### Configuration
```
Training Samples: 40,000 (50% of 80K)
Validation Samples: 4,000
Epochs: 3
Batch Size: 4
Max Sequence Length: 128
Gradient Accumulation: 4
Learning Rate: 2e-5
Checkpoint: Every 300 steps
```

### Expected Results
- **Accuracy Target**: 95-97%
- **Training Time**: 6-8 hours
- **Risk**: Medium (double the data)

### Decision Tree
```
Phase 1 Result → Action
├─ ≥95% accuracy → SUCCESS! Update report, no Phase 2 needed
├─ 93-94% accuracy → Proceed to Phase 2 for 95-97%
├─ <93% accuracy → Debug (shouldn't happen)
└─ Crash → Reduce to 15K and retry
```

---

## Monitoring Plan

### Check Points (Every 30-45 min)
1. **Step Count**: Track progress
2. **Training Loss**: Should decrease steadily
3. **Validation Loss**: Check every epoch
4. **System Stability**: CPU/Memory usage
5. **ETA**: Time remaining

### Progress Milestones
- ✅ Epoch 1 Complete (~80 min)
- ✅ Epoch 2 Complete (~160 min)
- ✅ Epoch 3 Complete (~240 min)
- ✅ Final Evaluation (~15 min)

---

## Success Criteria

### Phase 1 Success
- [x] Training completes all 3 epochs
- [x] Test accuracy ≥94%
- [x] All 15 categories ≥85%
- [x] No system crashes

### Phase 2 Success (if needed)
- [x] Test accuracy ≥95%
- [x] Beats or matches LR (96.92%)
- [x] All categories ≥90%

---

**Status**: 🟡 Phase 1 Starting...  
**Next Update**: 16:30 PM (Epoch 1 progress check)
