# from __future__ import print_function
# from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
# from pymavlink import mavutil # Needed for command message definitions
import time
import math
from Bebop import Bebop
# from pynput.keyboard import Controller
# # import time
#
bebop = Bebop()
def move(offset_x, offset_y, offset_z=0, time):
    # keyboard = Controller()
    # print(offset_x, offset_y)
    ''' text_file = open("buffer.txt", "w")

    target_y = 1
    target_x = 0
    if offset_y - target_y > 0:
        text_file.write("s")

    elif offset_y - target_y < 0:
        text_file.write("w")


    if offset_x - target_x > 0:
        text_file.write("a")
    elif offset_x - target_x < 0:
        text_file.write("d") '''
        bebop.fly_direct(roll=(100 * offset_y / abs(offset_y)), pitch=(100 * offset_x / abs(offset_x)), yaw=0, vertical_movement=(50 * offset_z/ abs(offset_z)), duration=time // 2)

def turn(direction, time):
    bebop.fly_direct(roll = 0, pitch = 0, yaw = 100 * direction, vertical_movement = 0, duration = time // 2)



    text_file.close()
def arm_and_takeoff(vehicle, aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    ''' while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)


    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)'''
    bebop.safe_takeoff(10)

def goto(vehicle, dNorth, dEast, gotoFunction=None):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for
    the target position. This allows it to be called with different position-setting commands.
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """

    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    #gotoFunction(targetLocation)
    print("Initial distance: ", targetDistance)
    gotoFunction(targetLocation)
    # print("location", currentLocation)
    # print("location", targetLocation)
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

    # while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
    #     #print "DEBUG: mode: %s" % vehicle.mode.name
    #     remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
    #     print("Distance to target: ", remainingDistance)
    #     if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
    #         print("Reached target")
    #         break;
    #     time.sleep(2)
    time.sleep(2)
    print("distance after move: ", get_distance_metres(vehicle.location.global_relative_frame, targetLocation))
def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")

    return targetlocation;


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5


def get_bearing(aLocation1, aLocation2):
    """
    Returns the bearing between the two LocationGlobal objects passed as parameters.

    This method is an approximation, and may not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    off_x = aLocation2.lon - aLocation1.lon
    off_y = aLocation2.lat - aLocation1.lat
    bearing = 90.00 + math.atan2(-off_y, off_x) * 57.2957795
    if bearing < 0:
        bearing += 360.00
    return bearing;
