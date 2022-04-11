# Makefile

EXE=d2q9-bgk

CC=icc
//CFLAGS= -std=c11 -Wall -g -O0
//CFLAGS= -std=c11 -Wall -Ofast -mtune=native -march=native -fopt-info-all=optRep.out -fopenmp -fopenmp-simd
CFLAGS = -std=c11 -Wall -fast -xHOST -qopenmp
//OtherIcc = -qopt-report=1 -qopt-report-phase=vec
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

check128:
	python check/check.py --ref-av-vels-file=check/128x128.av_vels.dat --ref-final-state-file=check/128x128.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat

check128x256:
	python check/check.py --ref-av-vels-file=check/128x256.av_vels.dat --ref-final-state-file=check/128x256.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat

check256:
	python check/check.py --ref-av-vels-file=check/256x256.av_vels.dat --ref-final-state-file=check/256x256.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat

check1024:
	python check/check.py --ref-av-vels-file=check/1024x1024.av_vels.dat --ref-final-state-file=check/1024x1024.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat

.PHONY: all check clean

clean:
	rm -f $(EXE)
