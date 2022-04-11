# Makefile

EXE=d2q9-bgk

BasicFlags = -std=c11 -Wall -Ofast -fopenmp
ReportFlags = -qopt-report=5 -g -shared-intel -D TBB_USE_THREADING_TOOLS -gline-tables-only -fdebug-info-for-profiling

CC=mpiicc
CFLAGS =  $(BasicFlags)
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/256x256.final_state.dat
REF_AV_VELS_FILE=check/256x256.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
