import numpy as np
from flex import irl
from flex.config import DELTA
from flex.editing import add, generalize, mask
from flex.features import feature_table, property_of
from flex.world import goal_index, start_index

def bic_score(features, context):
    # how well the features explain the demos, minus a penalty per feature
    table = feature_table(features, context.env)
    _, log_likelihood = irl.learn_weights(table, context.demos, start_index(context.env), goal_index(context.env))
    return log_likelihood - 0.5 * table.shape[1] * np.log(len(context.demos))

def flex_v(features, context):
    specific = add(features, context)
    general = generalize(specific, context)

    specific_props = set(property_of(f, context.env) for f in specific)
    general_props = set(property_of(f, context.env) for f in general)
    if specific_props != general_props:
        specific_score = bic_score(specific, context)
        general_score = bic_score(general, context)
        if abs(general_score - specific_score) < DELTA:
            # too close to call: ask the (simulated) user, who knows what they meant
            keep = context.scenario.truth in (general_props - specific_props)
            decision = 'asked'
        else:
            keep = general_score > specific_score
            decision = 'demos'
        context.log.append({'tag': 'verify', 'll_specific': round(float(specific_score), 2),
                            'll_general': round(float(general_score), 2),
                            'decision': decision, 'kept_generalization': bool(keep)})
        if not keep:
            general = specific
    return mask(general, context)
