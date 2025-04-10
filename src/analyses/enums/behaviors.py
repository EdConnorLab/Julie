from enum import Enum


class AgonisticBehaviors(Enum):
    NON_CONTACT_AGGRESSION = "non-contact aggression"
    RESOURCE_TAKEOVER = "resource takeover"
    MILD_AGGRESSION = "mild aggression"
    SEVERE_AGGRESSION = "severe aggression"


class SubmissiveBehaviors(Enum):
    SBT = "silent bared-teeth display"
    TURN_AWAY = "turn away"
    MOVE_AWAY = "move away"
    RUN_AWAY_SHORT = "run away short"
    RUN_AWAY_LONG = "run away long"
    CROUCH_NO_CONTACT = "crouch no contact"
    CROUCH_CONTACT = "crouch contact"


class AffiliativeBehaviors(Enum):
    GIVE_GROOM = "give groom"
    RECEIVE_GROOM = "receive groom"
    SOCIAL_PLAY = "social play"
    CONTACT = "contact"
    PROXIMITY = "proximity"


class IndividualBehaviors(Enum):
    LOCOMOTION = "locomotion"
    FEED_FORAGE = "feed forage"
    DRINK = "drink"
    INACTIVE_ALONE = "inactive alone"
    TOY = "toy"
    STAFF_INTERACTION = "staff interaction"
    OTHER = "other"
    UNKNOWN = "unknown"
