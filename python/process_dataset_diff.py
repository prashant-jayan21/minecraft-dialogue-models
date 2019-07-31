import json, itertools, pprint
from collections import Counter

with open("out-train.json") as json_data:
    results = json.load(json_data)

results_per_dialog = results["results"]

all_diff_items = list(map(
    lambda x: list(map(
        lambda y: {
            "diff_sample_keys": y["diff_sample_keys"],
            "diff_item_keys": y["diff_item_keys"],
            "old_sample_id": y["old_sample_id"],
            "new_sample_id": y["new_sample_id"],
            "json_id": x["json_id"],
            "logfile_path": x["logfile_path"]
        },
        x["diff_items"]
    )),
    results_per_dialog
))

all_diff_items = [ y for x in all_diff_items for y in x ]

# pprint.PrettyPrinter(indent=4).pprint(all_diff_items)
# print("\n\n")

counter = Counter()
for diff_item in all_diff_items:
    cause_effect = (tuple(diff_item["diff_sample_keys"]), tuple(diff_item["diff_item_keys"]))
    counter[cause_effect] += 1

print("CAUSE EFFECT DISTRIBUTION:")
pprint.PrettyPrinter(indent=4).pprint(counter.most_common())

groups = []
unique_cause_effects = []
all_diff_items = sorted(all_diff_items, key=lambda x: (x["diff_sample_keys"], x["diff_item_keys"]))
for k, g in itertools.groupby(all_diff_items, key=lambda x: (x["diff_sample_keys"], x["diff_item_keys"])):
    groups.append(list(g))
    unique_cause_effects.append(k)

# pprint.PrettyPrinter(indent=4).pprint(groups)
# print("\n\n")
pprint.PrettyPrinter(indent=4).pprint(unique_cause_effects)
print("\n\n")

# ad hoc stuff
pprint.PrettyPrinter(indent=4).pprint(groups[-1])
print("\n\n")
