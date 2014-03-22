__author__ = 'matt'

STR_INT_TITLE = 'title'
STR_INT_DESC = 'description'
STR_INT_RELATES = 'relates'

CLASSES_INT_CCW_I = {
    0: {
        STR_INT_TITLE: 'No relevant data',
        STR_INT_DESC: 'There is no data available for classification.',
        STR_INT_RELATES: [0]
    },
    1: {
        STR_INT_TITLE: 'Fine Sands',
        STR_INT_DESC: 'Mostly fine sands, textural ripples can be seen.',
        STR_INT_RELATES: [18, 19, 24]
    },
    2: {
        STR_INT_TITLE: 'Coarse sands with occasional rocks and fauna',
        STR_INT_DESC: 'Predominately course sands, rough texture with occasional fauna',
        STR_INT_RELATES: [16, 28, 29, 31]
    },
    3: {
        STR_INT_TITLE: 'Pebbled seabed with occasional rocks',
        STR_INT_DESC: 'Fairly consistent cobbled-esque texture of pebbles lining seabed.',
        STR_INT_RELATES: [20, 22]
    },
    4: {
        STR_INT_TITLE: 'Predominately large boulders',
        STR_INT_DESC: 'Area of sand or pebbles which is dominated by frequent boulders',
        STR_INT_RELATES: [14]
    },
    5: {
        STR_INT_TITLE: 'Coral & rich organic life',
        STR_INT_DESC: 'Numerous organisms such as starfish, sponges etc.',
        STR_INT_RELATES: [17, 23, 27]
    }
}


def get_internal_class_for_ccw(klass):
    """
    Utility function to map a classification from CCW schema to INT_CCW_I

    Returns int for class number in CLASSES_INT_CCW_I where klass is referenced.
    Returns None if none found.
    """
    d = CLASSES_INT_CCW_I
    klass = int(klass)

    for k in d:
        if klass in d[k][STR_INT_RELATES]:
            return k

    return None
