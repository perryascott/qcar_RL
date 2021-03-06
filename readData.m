

%read in ROS bag data to workspace
%bagID = "combined/combined3";

bag = rosbag('./bagFile_data/criticValueTest.bag');

position = select(bag,'Topic','/mocap/rigid_bodies/RigidBody_01/pose');
opti_msgs = readMessages(position,'DataFormat','struct');

x = cellfun(@(m) double(m.Pose.Position.X),opti_msgs);
y = cellfun(@(m) double(m.Pose.Position.Y),opti_msgs);
z = cellfun(@(m) double(m.Pose.Position.Z),opti_msgs);
time_sec = cellfun(@(m) double(m.Header.Stamp.Sec),opti_msgs);
time_nsec = cellfun(@(m) double(m.Header.Stamp.Nsec),opti_msgs);
pose_time = time_sec + time_nsec*10^(-9);
pose_bias = pose_time(1);
pose_time = pose_time - pose_bias; %extra .01 for optitrack delay

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

critic_topic = select(bag,'Topic','/critic_value');
msgs = readMessages(critic_topic,'DataFormat','struct');
long_act= cellfun(@(m) double(m.Vector.X),msgs);
steer_act= cellfun(@(m) double(m.Vector.Y),msgs);
critic_value= cellfun(@(m) double(m.Vector.Z),msgs);
time_sec = cellfun(@(m) double(m.Header.Stamp.Sec),msgs);
time_nsec = cellfun(@(m) double(m.Header.Stamp.Nsec),msgs);
crit_time = time_sec + time_nsec*10^(-9);
crit_time = crit_time - pose_bias;
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
    speed_time(i) = pose_time(i) + dT/2;
end
speed = (x_vel.^2 + y_vel.^2 + z_vel.^2).^(1/2);


x_vel_samp = zeros(length(x)-1,1);
y_vel_samp = zeros(length(y)-1,1);
z_vel_samp = zeros(length(z)-1,1);
speed_time_samp = zeros(length(pose_time)-1,1);

period = .2;
disc_time = .2;
prev_index = 1;
arr_index = 1
for i=1:length(x)-1
    if(pose_time(i+1) > disc_time)
        disc_time = disc_time + period;
        
    
        x_vel_samp(arr_index) = (x(i+1)-x(prev_index))/.2;
        y_vel_samp(arr_index) = (y(i+1)-y(prev_index))/.2;
        z_vel_samp(arr_index) = (z(i+1)-z(prev_index))/.2;
        speed_time_samp(arr_index) = pose_time(i+1);
        prev_index = i;
        arr_index = arr_index + 1;
    end
end
speed_samp = (x_vel_samp.^2 + y_vel_samp.^2 + z_vel_samp.^2).^(1/2);

speed_samp = speed_samp(1:find(speed_samp,1,'last'))
speed_time_samp = speed_time_samp(1:find(speed_time_samp,1,'last'))


close all
figure(1)

yyaxis left
plot(speed_time_samp,speed_samp)
ylabel('5Hz Sampled Speed State [m/s]');
ylim([-.5 2])

yyaxis right
plot(command_time,long_command)
ylabel('Throttle Command []');
%ylim([-40 40])
xlabel('Time [s]')
title("Throttle Command and Sampled State vs. Time")

figure(2)

yyaxis left
plot(speed_time_samp,abs(speed_samp))
yline(.7)
ylabel('5Hz Sampled Speed State [m/s]');
ylim([0 2.5])

yyaxis right
plot(crit_time,critic_value,"--")
ylabel('Critic Value');
ylim([-40 40])
xlabel('Time [s]')
title("Critic Model Value Output at 5Hz Sampled State")


figure(3)
%plot(speed_time,lowpass(speed,5,100))
plot(speed_time,speed)
hold on
plot(speed_time_samp,speed_samp)
title("Differing Sampling Time Speed vs. Time")
xlabel("Time [s]")
ylabel("Speed [m/s]")
legend(["Velocity Sampled at 100Hz", "Velocity Sampled at 5 Hz"])

