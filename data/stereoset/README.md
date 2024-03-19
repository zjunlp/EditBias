Thanks for [StereoSet](https://aclanthology.org/2021.acl-long.416/) from [bias-bench](https://github.com/McGill-NLP/bias-bench/tree/main/data/stereoset). 



- **train.json**, **dev.json**: 8:1 of test.json in StereoSet, including gender, race, and religion bias
- **gender_test.json**, **race_test.json**, **religion_test.json**: samples for gender, race, and religion bias from dev.json in StereoSet.
- **gender_test_reverse.json**: reversing the bias attribute words in `gender_test.json`
- **syn**: synonym-augmented `gender_test.json`, `race_test.json`, `religion_test.json`
