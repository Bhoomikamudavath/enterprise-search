import xml.etree.ElementTree as ET

for event, elem in ET.iterparse("data/Posts.xml", events=("end",)):
    if elem.tag == "row" and elem.get("PostTypeId") == "1":
        print(repr(elem.get("Tags")))
        break