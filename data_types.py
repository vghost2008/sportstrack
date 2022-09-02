import math

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4

class Angle(object):
    delta = 1e-3
    def __init__(self,_sentinel=None,angle=None,x=None,y=None) -> None:
        if angle is not None:
            self.angle = angle
        else:
            angle = math.atan2(y,x)
            if angle<0:
                angle += math.pi*2
            self.angle = angle*180/math.pi

    def __eq__(self,other):
        if math.fabs(self.angle-other.angle)<self.delta:
            return True
        else:
            if self.angle<other.angle:
                return math.fabs(self.angle+360-other.angle)<self.delta
            else:
                return math.fabs(self.angle-360-other.angle)<self.delta
        