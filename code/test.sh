#!/bin/bash

NET="fc4"
TEST_IDX="1"
EPS="0.09000"

echo Verifying ${NET} with img ${TEST_IDX} and eps ${EPS}
python verifier.py --net "${NET}" --spec "../test_cases/${NET}/img${TEST_IDX}_${EPS}.txt"
