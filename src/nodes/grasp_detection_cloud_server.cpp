#include <nodes/grasp_detection_cloud_server.h>


GraspDetectionCloudServer::GraspDetectionCloudServer(ros::NodeHandle& node)
{
  cloud_camera_ = NULL;

  // set camera viewpoint to default origin
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  view_point_ << camera_position[0], camera_position[1], camera_position[2];

  grasp_detector_ = new GraspDetector(node);

  std::string rviz_topic;
  node.param("rviz_topic", rviz_topic, std::string(""));

  if (!rviz_topic.empty())
  {
    rviz_plotter_ = new GraspPlotter(node, grasp_detector_->getHandSearchParameters());
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  // Advertise ROS topic for detected grasps.
  grasps_pub_ = node.advertise<gpd::GraspConfigList>("clustered_grasps", 10);

  node.getParam("workspace", workspace_);
}


bool GraspDetectionCloudServer::detectGrasps(gpd::detect_grasps_cloud::Request& req, gpd::detect_grasps_cloud::Response& res)
{
  ROS_INFO("Received service request ...");

  // 1. Initialize cloud camera.
  cloud_camera_ = NULL;
  const sensor_msgs::PointCloud2 msg = req.cloud;

    // Set view points.
  Eigen::Matrix3Xd view_points(3,1);
  view_points.col(0) = view_point_;

  if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
      && msg.fields[5].name == "normal_z")
    {
      PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
    }
    else
    {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
    }


    frame_ = msg.header.frame_id;

  // 2. Preprocess the point cloud.
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

  // 3. Detect grasps in the point cloud.
  std::vector<Grasp> grasps = grasp_detector_->detectGrasps(*cloud_camera_);

  if (grasps.size() > 0)
  {
    // Visualize the detected grasps in rviz.
    if (use_rviz_)
    {
      rviz_plotter_->drawGrasps(grasps, frame_);
    }

    // Publish the detected grasps.
    gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
    res.grasp_configs = selected_grasps_msg;
    ROS_INFO_STREAM("Detected " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");
    return true;
  }

  ROS_WARN("No grasps detected!");
  return false;
}

bool GraspDetectionCloudServer::detectGraspsInMsg(const sensor_msgs::PointCloud2& msg, gpd::GraspConfigList& gpd_grasps)
{
  ROS_INFO("Received cloud msg");

  // 1. Initialize cloud camera.
  cloud_camera_ = NULL;

    // Set view points.
  Eigen::Matrix3Xd view_points(3,1);
  view_points.col(0) = view_point_;

  if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
      && msg.fields[5].name == "normal_z")
    {
      PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
    }
    else
    {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
    }


    frame_ = msg.header.frame_id;

  // 2. Preprocess the point cloud.
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

  // 3. Detect grasps in the point cloud.
  std::vector<Grasp> grasps = grasp_detector_->detectGrasps(*cloud_camera_);

  if (grasps.size() > 0)
  {
    // Visualize the detected grasps in rviz.
    if (use_rviz_)
    {
      rviz_plotter_->drawGrasps(grasps, frame_);
    }

    // Publish the detected grasps.
    gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
    ROS_INFO_STREAM("Detected " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");
    gpd_grasps = selected_grasps_msg;

    return true;
  }

  ROS_WARN("No grasps detected!");
  return false;
}


gpd::GraspConfigList GraspDetectionCloudServer::createGraspListMsg(const std::vector<Grasp>& hands)
{
  gpd::GraspConfigList msg;

  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(convertToGraspMsg(hands[i]));

  msg.header = cloud_camera_header_;

  return msg;
}


gpd::GraspConfig GraspDetectionCloudServer::convertToGraspMsg(const Grasp& hand)
{
  gpd::GraspConfig msg;
  tf::pointEigenToMsg(hand.getGraspBottom(), msg.bottom);
  tf::pointEigenToMsg(hand.getGraspTop(), msg.top);
  tf::pointEigenToMsg(hand.getGraspSurface(), msg.surface);
  tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
  tf::vectorEigenToMsg(hand.getBinormal(), msg.binormal);
  tf::vectorEigenToMsg(hand.getAxis(), msg.axis);
  msg.width.data = hand.getGraspWidth();
  msg.score.data = hand.getScore();
  tf::pointEigenToMsg(hand.getSample(), msg.sample);

  return msg;
}


int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "detect_grasps_server");
  ros::NodeHandle node("~");

  GraspDetectionCloudServer grasp_detection_server(node);

  ros::ServiceServer service = node.advertiseService("detect_grasps", &GraspDetectionCloudServer::detectGrasps,
                                                     &grasp_detection_server);
  ROS_INFO("Grasp detection service is waiting for a point cloud ...");

  ros::spin();

  return 0;
}
