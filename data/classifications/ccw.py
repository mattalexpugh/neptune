__author__ = 'matt'

"""
This dictionary has been adapted from the document:

Groups of marine habitats found around the coasts and sea of
Wales for use in assessing sensitivity to different fishing
activities

Created by the Countryside Council for Wales (CCW)
For further information contact Dr Kirsten Ramsay k.ramsay@ccw.gov.uk

We are specifically interested in classifications:

    16, 18, 19, 23, 24, 27*, 28* and 31*

* May not be entirely relevant

Dictionary of order:

    id : int => {
        title: str,
        description: str
    }

"""

STR_CCW_TITLE = 'title'
STR_CCW_DESC = 'description'

CLASSES_CCW = {
    0: {
        STR_CCW_TITLE: 'No relevant data',
        STR_CCW_DESC: "There is no data available for classification."
    },
    14: {
        STR_CCW_TITLE: 'Vertical subtidal rock with associated community' +
                       'include large and/or long lived bivalves',
        STR_CCW_DESC: """A diverse array of communities can be associated with vertical rock; typically assemblages of sponges, hydroids, bryozoans and anemones."""
    },
    16: {
        STR_CCW_TITLE: 'Coarse sands and gravels with communities that ' +
                       'include large and/or long lived bivalves',
        STR_CCW_DESC: """Subtidal habitats known to support long-lived
bivalves such as Venerids. These bivalves are
particularly vulnerable to disturbance because they
are long-lived (oldest specimen found in the Irish
Sea was 140 years old) and take a long time to reach
reproductive maturity."""
    },
    17: {
        STR_CCW_TITLE: 'Maerl beds' +
                       'include large and/or long lived bivalves',
        STR_CCW_DESC: """Maerl is a calcareous red algae that can form dense beds that support particularly diverse communities. It is a BAP habitat and species, and it is nationally scarce."""
    },
    18: {
        STR_CCW_TITLE: 'Stable predominantly subtidal fine sands',
        STR_CCW_DESC: """Clean fine sands with less than 5% silt/clay in
deeper water, either on the open coast or in tide-
swept channels of marine inlets in depths of over
15-20m. This habitat is generally more stable than
shallower, infralittoral sands and consequently
supports a more diverse community. Includes
biotopes with dense aggregations of the polychaete
Lanice concheliga."""
    },
    19: {
        STR_CCW_TITLE: 'Subtidal stable muddy sands, sandy muds and muds',
        STR_CCW_DESC: """A wide variety of stable sediment biotopes
supporting animal dominated communities."""
    },
    20: {
        STR_CCW_TITLE: 'Predominantly subtidal rock with low-lying and fast growing faunal turf',
        STR_CCW_DESC: """Rock dominated by a variety of low-lying faunal turf forming organisms such as bryozoans and hydroids. This habitat mainly occurs on extremely exposed to moderately wave-exposed circalittoral bedrock and boulders.
Often species rich, their low-lying form means that they may be less vulnerable to physical disturbance than areas that support larger, erect species (i.e. habitat 15 & 16)."""
    },
    22: {
        STR_CCW_TITLE: 'Shallow subtidal rock with kelp',
        STR_CCW_DESC: """Diverse range of generally species-rich communities occurring on shallow subtidal rock. This habitat is characterised by kelps but also includes red foliose seaweeds and surge gullies."""
    },
    23: {
        STR_CCW_TITLE: 'Kelp and seaweed communities on sand scoured rock',
        STR_CCW_DESC: """Sediment-affected or disturbed kelp and seaweed
communities and kelp and seaweed communities on
sediment. This includes rock habitats subject to
scour by mobile sediments from nearby areas. The
associated communities can be quite variable in
character, depending on the particular conditions
which prevail."""
    },
    24: {
        STR_CCW_TITLE: 'Dynamic, shallow water fine sands',
        STR_CCW_DESC: """Clean sands which occur in shallow water, either on
the open coast or in tide-swept channels of marine
inlets. The habitat is characterised by robust fauna,
particularly amphipods (Bathyporeia) and robust
polychaetes including Nephtys cirrosa and Lanice
conchilega. This group also includes mobile mud
biotopes."""
    },
    27: {
        STR_CCW_TITLE: 'Biogenic reef on sediment and mixed substrata',
        STR_CCW_DESC: """Certain marine species, such as mussels and some
worms, can occur in very dense aggregations on the
sea bed. Such biogenic reefs tend to stabilize the
sediment and provide a physical structure that can
supports diverse assemblages of other organisms.
Key examples are Horse mussel reefs, mussel beds,
and the subtidal honeycomb worm."""
    },
    28: {
        STR_CCW_TITLE: 'Stable, species rich mixed sediments',
        STR_CCW_DESC: """These habitats incorporate a range of sediments
including heterogeneous muddy gravelly sands and
also mosaics of cobbles and pebbles embedded in or
lying upon sand, gravel or mud. These habitats tend
to be stable and species rich, supporting a wide
range of organisms both within, and on the seabed
including worms, bivalves, echinoderms, anemones,
hydroids and bryozoa. Burrowing crustacea are also
often a feature of this habitat , e.g. Upogebia
deltaura and Rissoides desmaresti."""
    },
    29: {
        STR_CCW_TITLE: 'Unstable cobbles, pebbles, gravels and/or coarse sands supporting relatively robust communities',
        STR_CCW_DESC: """Coarse sediments including coarse sand, gravel, pebbles, shingle and cobbles which are often unstable due to tidal currents and/or wave action. These habitats are generally found on the open coast or in tide-swept channels of marine inlets. They typically have a low silt content and lack a significant seaweed component. They are characterised by a robust fauna."""
    },
    31: {
        STR_CCW_TITLE: 'Stable but generally tideswept cobbles, ' +
                       'pebbles, gravels and coarse sediments',
        STR_CCW_DESC: """Coarse sediments including coarse sand, gravel,
pebbles, shingle and cobbles which are often
compacted together and may be described as lag
pavements (areas of glacial deposits where the finer
sediments have subsequently been eroded). These are
characterised by epifaunal species such as the
bryozoan Flustra foliacea and other hydroids and
bryozoans."""
    }
}
