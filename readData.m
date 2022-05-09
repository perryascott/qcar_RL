

%read in ROS bag data to workspace
%bagID = "combined/combined3";

bag = rosbag('delayTest.bag');

position = select(bag,'Topic','/mocap/rigid_bodies/RigidBody_01/pose');
opti_msgs = readMessages(position,'DataFormat','struct');

x = cellfun(@(m) double(m.Pose.Position.X),opti_msgs);
y = cellfun(@(m) double(m.Pose.Position.Y),opti_msgs);
z = cellfun(@(m) double(m.Pose.Position.Z),opti_msgs);
time_sec = cellfun(@(m) double(m.Header.Stamp.Sec),opti_msgs);
time_nsec = cellfun(@(m) double(m.Header.Stamp.Nsec),opti_msgs);
pose_time = time_sec + time_nsec*10^(-9);
pose_bias = pose_time(1);
pose_time = pose_time - pose_bias;

odom = select(bag,'Topic','/qcar/odometry');
msgs = readMessages(odom,'DataFormat','struct');
raw_odom = cellfun(@(m) double(m.Vector.X),msgs);
time_sec = cellfun(@(m) double(m.Header.Stamp.Sec),msgs);
time_nsec = cellfun(@(m) double(m.Header.Stamp.Nsec),msgs);
odom_time = time_sec + time_nsec*10^(-9);
odom_time = odom_time - pose_bias;

accel = select(bag,'Topic','/qcar/accelerometer');
msgs = readMessages(accel,'DataFormat','struct');
raw_accel = cellfun(@(m) double(m.Vector.X),msgs);
y_acc_r = cellfun(@(m) double(m.Vector.Y),msgs);
z_acc_r = cellfun(@(m) double(m.Vector.Z),msgs);
time_sec = cellfun(@(m) double(m.Header.Stamp.Sec),msgs);
time_nsec = cellfun(@(m) double(m.Header.Stamp.Nsec),msgs);
accel_time = time_sec + time_nsec*10^(-9);
accel_time = accel_time - pose_bias;

command = select(bag,'Topic','/qcar/user_command');
msgs = readMessages(command,'DataFormat','struct');
long_command= cellfun(@(m) double(m.Vector.X),msgs);
time_sec = cellfun(@(m) double(m.Header.Stamp.Sec),msgs);
time_nsec = cellfun(@(m) double(m.Header.Stamp.Nsec),msgs);
command_time = time_sec + time_nsec*10^(-9);
command_time = command_time - pose_bias;

%how far behind the velocity is behind accelerometer
opti_lag = .01
pose_time = pose_time -opti_lag

%% calculate velocity

x_vel = zeros(length(x)-1,1);
y_vel = zeros(length(y)-1,1);
z_vel = zeros(length(z)-1,1);
speed_time = zeros(length(pose_time)-1,1);

Ts = 1/100;


for i=1:length(x)-1
    dT = pose_time(i+1)-pose_time(i);
    x_vel(i) = (x(i+1)-x(i))/dT;
    y_vel(i) = (y(i+1)-y(i))/dT;
    z_vel(i) = (z(i+1)-z(i))/dT;
    speed_time(i) = pose_
    time(i) + dT/2;
end
speed = (x_vel.^2 + y_vel.^2 + z_vel.^2).^(1/2);

figure(1)
plot(speed_time,speed)
hold on
plot(command_time,long_command)
legend(["Velocity", "Throttle Command"])



