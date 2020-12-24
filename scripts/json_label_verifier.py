import json
import logging
import sys


def verify_labeling(data):
    allowed_classes = ["Paprika", "Kiwi"]
    allowed_region_attributes = ["Objects"]
    errors = 0

    for key, image_description in data.items():
        contained_classes = []
        regions = image_description['regions']
        if len(regions) == 0:
            logging.warning(f"No regions in image {key}.")
            errors += 1

        for i, region in enumerate(regions):
            region_attributes = region['region_attributes']
            if len(region_attributes) == 0:
                logging.warning(f"No region attributes in region {i} of image {key} "
                                "(this might indicate a missing label)")
                errors += 1
                continue
            elif len(region_attributes) > 1:
                logging.warning(f"More than one region attribute ({len(region_attributes)}) "
                                f"in region {i} of image {key}")
                errors += 1
                continue

            for attribute in region_attributes:
                if attribute not in allowed_region_attributes:
                    logging.warning(f"Invaid region attribute {region_attributes[0]} "
                                    f"in region {i} of image {key}")
                    errors += 1
                    continue

            classification = region_attributes["Objects"]
            if classification not in allowed_classes:
                logging.warning(f"Invaid class {classification} in region {i} of image {key}")
                errors += 1
                continue
            contained_classes.append(classification)

        expected_classes = []
        if "kiw" in key:
            expected_classes.append("Kiwi")
        if "pap" in key:
            expected_classes.append("Paprika")

        if set(expected_classes) != set(contained_classes):
            logging.warning("Labels do not conform to the file name: expected:"
                            f"{set(expected_classes)} but found: {set(contained_classes)}")
            errors += 1
    return errors


def verify_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
        errors = verify_labeling(data)

        if errors > 0:
            logging.info(f"{errors} errors found.")
        else:
            logging.info("Success, no errors found.")


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Please give a path to the JSON as command line argument.")
    else:
        verify_json(sys.argv[1])
