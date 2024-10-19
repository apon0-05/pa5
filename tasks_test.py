import json
import os
import unittest

from upload_reports import report_file


# DO NOT CHANGE this file

def print_score(score: int, task: str):
    print(f'\nScore for {task}: {score} out of 100\n')


class TestTasks(unittest.TestCase):

    def test_pa5(self):
        os.system("jupyter nbconvert --to python pa5.ipynb")
        import pa5
        a = pa5.score_a
        b = pa5.score_b
        c = pa5.score_c
        print_score(a, 'A')
        print_score(b, 'B')
        print_score(c, 'C')

        try:
            with open(report_file, 'r') as infile:
                scores = json.load(infile)
        except Exception:
            scores = {}

        scores['a'] = a
        scores['b'] = b
        scores['c'] = c

        json_object = json.dumps(scores, indent=4)
        with open(report_file, 'w+') as outfile:
            outfile.write(json_object)

        os.remove('./pa5.py')


if __name__ == "__main__":
    unittest.main()
