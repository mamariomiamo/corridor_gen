#include <corridor_gen.h>

using namespace CorridorGen;

CorridorGenerator::CorridorGenerator(double resolution, double clearance, int max_sample, double ceiling, double floor, double closeness_threshold, double desired_radius, std::array<double, 4> weighting)
    : resolution_(resolution)
    , clearance_(clearance)
    , max_sample_(max_sample)
    , ceiling_(ceiling)
    , floor_(floor)
    , closeness_threshold_(closeness_threshold)
    , desired_radius_(desired_radius)
    , weighting_(weighting)
{
    octree_.deleteTree();
    octree_.setResolution(resolution_);
    std::random_device rd;
    gen_ = std::mt19937_64(rd());
    one_third_ = 1.0 / 3.0;
    drone_wheelbase_ = desired_radius_/4;
    closeness_threshold_lb_ = drone_wheelbase_;
    closeness_threshold_ = desired_radius_;
}

CorridorGenerator::CorridorGenerator(double resolution, double clearance, int max_sample, double ceiling, double floor, double closeness_threshold, double desired_radius)
    : resolution_(resolution)
    , clearance_(clearance)
    , max_sample_(max_sample)
    , ceiling_(ceiling)
    , floor_(floor)
    , closeness_threshold_(closeness_threshold)
    , desired_radius_(desired_radius)
{
    octree_.deleteTree();
    octree_.setResolution(resolution_);
    std::random_device rd;
    gen_ = std::mt19937_64(rd());
    one_third_ = 1.0 / 3.0;
    drone_wheelbase_ = desired_radius_/4;
    closeness_threshold_lb_ = drone_wheelbase_;
    closeness_threshold_ = desired_radius_;
}

void CorridorGenerator::setNoFlyZone(std::vector<Eigen::Vector4d> no_flight_zone)
{
    no_flight_zone_ = no_flight_zone;
}

void CorridorGenerator::updatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &new_cloud)
{
    if (new_cloud->points.empty())
    {
        cloud_empty_ = true;
        return;
    }
    else
    {
        cloud_empty_ = false;
        cloud_ = new_cloud;
        octree_.deleteTree();
        octree_.setInputCloud(new_cloud);
        octree_.addPointsFromInputCloud();
    }

    // ToDo how to check octree.leaf_count_ >= 0 ?
}

void CorridorGenerator::updateGlobalPath(const std::vector<Eigen::Vector3d> &path)
{
    guide_path_.clear();
    guide_path_ = path;               // copy assignment
    initial_position_ = path.front(); // front() and back() return read-only reference
    goal_position_ = path.back();
    std::reverse(guide_path_.begin(), guide_path_.end());
}

bool CorridorGenerator::generateCorridorAlongPath(const std::vector<Eigen::Vector3d> &path)
{
    flight_corridor_.clear(); // clear flight corridor generated from previous iteration
    waypoint_list_.clear();   // clear waypoints initialized from previous iteration
    sample_direction_.clear();
    updateGlobalPath(path);
    waypoint_list_.push_back(initial_position_);
    Corridor current_corridor = GenerateOneSphere(initial_position_);
    Corridor previous_corridor = current_corridor;
    flight_corridor_.emplace_back(current_corridor);
    int max_iter = guide_path_.size();
    int iter_count = 0;
    bool guide_point_adjusted = false;
    Eigen::Vector3d sample_origin;
    int stagnant_count = 0;
    while (true)
    {
        iter_count++;
        if (iter_count > 200)
        {
            std::cout << "OVER MAX_ITER FAIL to generate corridor" << std::endl;
            return false;
            // break;
        }
        if (!guide_point_adjusted)
        {
            stagnant_count = 0;
            local_guide_point_ = getGuidePoint(guide_path_, current_corridor);
            sample_origin = local_guide_point_;
        }

        if (bash_through_)
        {
            next_local_guide_point_ = guide_path_.end()[-2]; // second last point of the path list
        }

        current_corridor = uniformBatchSample(local_guide_point_, next_local_guide_point_, current_corridor, sample_origin, guide_point_adjusted, bash_through_);
        double distance_between_corridor_center = (current_corridor.first - previous_corridor.first).norm();

        // check if the corridor did not move much i.e. clutter place
        // need to adjust guide point according?
        // guide point will affect the sample direction
        // AND sample origin
        // after adjustment, we will need to use the same sample origin?

        if (corridorTooClose(current_corridor)) // todo: this can be checked at the start?
        {
            guide_point_adjusted = true;
            current_corridor = previous_corridor; // is this needed?
            stagnant_count++;
            closeness_threshold_ = current_corridor.second;
            if (guide_path_.size() < 2)
            {
                std::cout << "TOO STAGNANT and guide_path empty FAIL to generate corridor" << std::endl;
                bash_through_ = true;
                return false;
                // break;
            }
            if (stagnant_count > 3) // adjust closeness_threshold
            {
                closeness_threshold_ = closeness_threshold_ - stagnant_count * 0.1;
                if (closeness_threshold_ < closeness_threshold_lb_)
                {
                    closeness_threshold_ = closeness_threshold_lb_;
                }

                std::cout << "TOO STAGNANT need to bash_through_" << std::endl;
                // we will not sample for now since this happends during extremely cluttered environment.
                // we will generate a sphere on the circumference of the current sphere
                // we point from the center of the sphere to the to same direction
                // as the vector from by guide point and the next guide point
                // so we can bash through the cluttered area
                bash_through_ = true;
                // break;
                stagnant_count = 0;
                // guide_path_.pop_back();
                // local_guide_point_ = guide_path_.back();
                // sample_origin = local_guide_point_;
            }

            else
            {
                guide_path_.pop_back();
                local_guide_point_ = guide_path_.back();
                // sample_origin = local_guide_point_;
                continue;
            }

            continue;
        }
        else
        {
            guide_point_adjusted = false;
            bash_through_ = false;
        }

        Eigen::Vector3d intermediate_waypoint = getMidPointBetweenCorridors(previous_corridor, current_corridor);
        previous_corridor = current_corridor;

        flight_corridor_.emplace_back(current_corridor);
        waypoint_list_.emplace_back(intermediate_waypoint);

        if (pointInCorridor(goal_position_, current_corridor))
        {
            std::cout << "Corridors Generated" << std::endl;
            waypoint_list_.emplace_back(goal_position_);
            return true;
        }
    }
}

Corridor CorridorGenerator::GenerateOneSphere(const Eigen::Vector3d &pos)
{
    // lock guard to prevent cloud from changing?
    int K = 1;
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    pcl::PointXYZ searchPoint;
    searchPoint.x = pos.x();
    searchPoint.y = pos.y();
    searchPoint.z = pos.z();

    if (!cloud_empty_)
    {
        if (octree_.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            // std::cout << "Found neighbor" << std::endl;
            double radius = sqrt(double(pointNKNSquaredDistance[0])) - clearance_;
            Eigen::Vector3d nearestPt;

            nearestPt.x() = (*cloud_)[pointIdxNKNSearch[0]].x;
            nearestPt.y() = (*cloud_)[pointIdxNKNSearch[0]].y;
            nearestPt.z() = (*cloud_)[pointIdxNKNSearch[0]].z;

            // std::cout << "nearestPt is " << nearestPt.transpose() << std::endl;

            if (radius > searchPoint.z)
            {
                radius = abs(searchPoint.z);
            }

            return Corridor(pos, std::min(desired_radius_, radius));
        }

        else
        {
            return Corridor(pos, desired_radius_);
        }
    }
    else
    {
        return Corridor(pos, desired_radius_);
    }
}

auto CorridorGenerator::GenerateOneSphereVerbose(const Eigen::Vector3d &pos)
{
    // lock guard to prevent cloud from changing?
    int K = 1;
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    pcl::PointXYZ searchPoint;
    searchPoint.x = pos.x();
    searchPoint.y = pos.y();
    searchPoint.z = pos.z();
    Eigen::Vector3d nearestPt;
    // consider using radiusSearch?

    // check if octree is empty
    // use flag to decide if search is needed
    if (!cloud_empty_)
    {
        if (octree_.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            double radius;

            if (bash_through_)
            {
                radius = sqrt(double(pointNKNSquaredDistance[0]));
            }
            else
            {
                radius = sqrt(double(pointNKNSquaredDistance[0])) - clearance_;
            }

            nearestPt.x() = (*cloud_)[pointIdxNKNSearch[0]].x;
            nearestPt.y() = (*cloud_)[pointIdxNKNSearch[0]].y;
            nearestPt.z() = (*cloud_)[pointIdxNKNSearch[0]].z;

            // std::cout << "nearestPt is " << nearestPt.transpose() << std::endl;

            if (radius > searchPoint.z)
            {
                radius = abs(searchPoint.z);
            }

            // return Corridor(pos, radius);
            return std::make_pair(Corridor(pos, radius), nearestPt);
        }

        else
        {
            nearestPt << 50, 50, 50; // Arbitrarily far value and it will not be used
            // return Corridor(pos, 10.0);
            return std::make_pair(Corridor(pos, desired_radius_), nearestPt);
        }
    }

    else
    {
        nearestPt << 50, 50, 50; // Arbitrarily far value and it will not be used
        return std::make_pair(Corridor(pos, desired_radius_), nearestPt);
    }
}

bool CorridorGenerator::CheckSingleCorridorSafety(const Corridor &corridor)
{
    auto [center, radius] = corridor;
    // std::cout << "old corridor center: " << center.transpose() << " radus: " << radius << std::endl;
    auto ret = GenerateOneSphereVerbose(center);
    auto [new_center, new_radius] = ret.first;
    // std::cout << "new corridor center: " << new_center.transpose() << " radus: " << new_radius << std::endl;
    if (new_radius < radius - 0.1)
    {
        return false; // Corridor is unsafe and should be invalidaed
    }
    else
    {
        return true; // Corridor is safe and valid
    }
}

std::tuple<bool, int, Eigen::Vector3d> CorridorGenerator::CheckCorridorSafety(const std::vector<Corridor> corridor_list)
{
    for (int i = 1; i < corridor_list.size(); i++) // we skip the check for the first corridor
    {
        if (CheckSingleCorridorSafety(corridor_list[i]))
        {
            continue;
        }
        else
        {
            int last_safe_corridor_index = i - 1;
            Eigen::Vector3d last_safe_corridor_center = corridor_list[last_safe_corridor_index].first;
            // std::cout << "corridor index: " << i << " is unsafe\n";
            // std::cout << "last safe corridor center is " << last_safe_corridor_center.transpose() << std::endl;
            return std::make_tuple(false, last_safe_corridor_index, last_safe_corridor_center);
        }
    }
    std::cout << "All corridor safe\n";
    return std::make_tuple(true, corridor_list.size() - 1, corridor_list.back().first);
}

Eigen::Vector3d CorridorGenerator::getGuidePoint(std::vector<Eigen::Vector3d> &guide_path, const Corridor &input_corridor)
{
    auto [center, radius] = input_corridor;

    while ((center - guide_path.back()).norm() < radius && guide_path.size() > 1)
    {
        guide_path.pop_back();
    }

    Eigen::Vector3d ret;
    ret = guide_path.back();

    return ret;
}

bool CorridorGenerator::pointInCorridor(const Eigen::Vector3d &point, const Corridor &corridor)
{
    auto [center, radius] = corridor;
    return ((point - center).norm()) < radius;
}

Corridor CorridorGenerator::uniformBatchSample(const Eigen::Vector3d &guide_point, const Eigen::Vector3d &guide_point_next, const Corridor &input_corridor, const Eigen::Vector3d &sample_around, bool guide_point_adjusted, bool bash_through)
{
    bool sample_in_clutter;
    auto [previous_center, pre_radius] = input_corridor;

    if (bash_through)
    {
        std::cout << "bashing through\n";
        Eigen::Vector3d direction = (guide_point_next - guide_point).normalized();
        Eigen::Vector3d new_center = previous_center + pre_radius * direction;
        // center of the sphere corridor is on the perimeter of the previous corridor
        auto ret = GenerateOneSphereVerbose(new_center);
        Corridor new_corridor = ret.first;
        return new_corridor;
    }

    else
    {
        Eigen::Vector3d sample_origin;
        if (guide_point_adjusted)
        {
            sample_in_clutter = true;
            Eigen::Vector3d sample_direction = guide_point - previous_center;
            sample_direction = sample_direction / sample_direction.norm();
            sample_origin = sample_direction * pre_radius + previous_center;
        }
        else
        {
            sample_in_clutter = false;
            sample_origin = sample_around;
        }

        // auto [pre_center, pre_radius] = input_corridor;
        // if guide_point is adjusted i.e. it will lead to early return
        // because there will be no overlap between the previous and new corridors
        // auto ret = GenerateOneSphereVerbose(guide_point);

        //
        auto ret = GenerateOneSphereVerbose(sample_origin);

        // std::cout << "sample_origin " << sample_origin.transpose() << std::endl;

        Corridor guide_pt_corridor = ret.first;
        Eigen::Vector3d obstacle = ret.second;
        auto [center, radius] = guide_pt_corridor;
        double resolution = 0.1;
        std::vector<Eigen::Vector3d> sample_point_bucket;

        // Following check is needed if guide_point is adjusted
        // Also handle Special case when the sample_origin (guide point) is very close to the obstacle
        // we will continue for batchsample
        double distance_between = (previous_center - center).norm();
        if (distance_between > (pre_radius + radius) && (radius > drone_wheelbase_) && (!guide_point_adjusted))
        {
            std::cout << "Candidate too far, no overlap " << center.transpose() << std::endl;
            return input_corridor;
        }

        if (distance_between > (pre_radius + radius) && (radius > drone_wheelbase_) && (guide_point_adjusted))
        {
            std::cout << "After adjustment Candidate too far, no overlap " << center.transpose() << std::endl;
            return input_corridor;
        }

        if (radius + resolution > desired_radius_)
        {
            return guide_pt_corridor;
        }

        else
        {
            Eigen::Vector3d sample_to;
            Eigen::Vector3d sample_from;
            sample_from = sample_origin;
            sample_to = previous_center;
            sample_to.z() = sample_from.z(); // to ensure sample direction is along the guide path z plane
            Eigen::Vector3d sample_x_vector = sample_to - guide_point;
            Eigen::Vector3d sample_dir_coord = sample_x_vector + guide_point;

            sample_direction_.push_back(std::make_pair(sample_from, sample_to)); // for visualization

            Eigen::Vector3d sample_x_direction = sample_x_vector / sample_x_vector.norm();
            Eigen::Vector3d global_x_direction(1, 0, 0);
            double angle_between = acos(sample_x_direction.dot(global_x_direction));
            Eigen::Vector3d rotation_axis;
            Eigen::Matrix3d rotation_matrix;

            double x_var = one_third_ * sample_x_vector.norm() / 10;
            double y_var = 2 * x_var;
            double z_var = y_var;
            double x_sd = sqrt(x_var);
            double y_sd = sqrt(y_var);
            double z_sd = y_sd;
            int sample_num = 0;

            if ((0 <= angle_between && angle_between < 0.017) || (0 >= angle_between && angle_between > -0.017))
            {
                rotation_matrix << 1, 0, 0, 0, 1, 0, 0, 0, 1;
            }

            else if ((3.124 <= angle_between && angle_between < 3.159) || (-3.124 >= angle_between && angle_between > -3.159))
            {
                rotation_matrix << -1, 0, 0, 0, -1, 0, 0, 0, 1;
            }

            else
            {
                rotation_axis = global_x_direction.cross(sample_x_direction);
                rotation_axis = rotation_axis / rotation_axis.norm();
                rotation_matrix = Eigen::AngleAxisd(angle_between, rotation_axis);
            }

            x_normal_rand_ = std::normal_distribution<double>(0, x_sd);
            y_normal_rand_ = std::normal_distribution<double>(0, y_sd);
            z_normal_rand_ = std::normal_distribution<double>(0, z_sd);
            double sample_length = sample_x_vector.norm();
            // std::cout << "-sample_length, sample_length " << sample_length << std::endl;
            std::uniform_real_distribution<double> x_uniform_rand_(-sample_length, sample_length);

            double y_val, z_val;
            y_val = drone_wheelbase_ * 2;
            z_val = drone_wheelbase_;

            std::uniform_real_distribution<double> y_uniform_rand_(-y_val, y_val);
            std::uniform_real_distribution<double> z_uniform_rand_(-z_val, z_val);

            std::priority_queue<CorridorPtr, std::vector<CorridorPtr>, CorridorComparator> empty_pool;
            corridor_pool_.swap(empty_pool);

            Eigen::Vector2d weighting;

            while (sample_num < max_sample_)
            {
                sample_num++;
                Eigen::Vector3d candidate_pt;
                if (sample_in_clutter)
                {
                    // std::cout << "sample_in_clutter" << std::endl;

                    x_uniform_rand_ = std::uniform_real_distribution<double>(-sample_length, 0);
                    y_uniform_rand_ = std::uniform_real_distribution<double>(-sample_length, sample_length);
                    z_uniform_rand_ = std::uniform_real_distribution<double>(0, 0);
                    // double x_coord = x_normal_rand_(gen_);
                    // double y_coord = y_normal_rand_(gen_);
                    // double z_coord = z_normal_rand_(gen_);
                    double x_coord = x_uniform_rand_(gen_);
                    double y_coord = y_uniform_rand_(gen_);
                    double z_coord = z_uniform_rand_(gen_);
                    candidate_pt << x_coord, y_coord, 0.0;
                    weighting << weighting_[0], weighting_[1]; // weight on ego_volume, overlap_volume, we prioritize overlapping volume
                }

                else
                {
                    double x_coord = x_normal_rand_(gen_);
                    double y_coord = y_normal_rand_(gen_);
                    double z_coord = z_normal_rand_(gen_);

                    // Eigen::Vector3d candidate_pt(0.0, y_coord, z_coord);
                    // Eigen::Vector3d candidate_pt(x_coord, y_coord, z_coord);

                    candidate_pt << x_coord, y_coord, z_coord;
                    weighting << weighting_[2], weighting_[3]; // weight on ego_volume, overlap_volume
                }

                // needs to transform the candidate_pt to align the axes
                candidate_pt = rotation_matrix * candidate_pt + sample_origin;
                candidate_pt.z() = std::max(
                    std::min(sample_origin.z(), ceiling_), floor_);

                pcl::PointXYZ candidate_pt_pcl;
                candidate_pt_pcl.x = candidate_pt.x();
                candidate_pt_pcl.y = candidate_pt.y();
                candidate_pt_pcl.z = candidate_pt.z();

                // check if candidate_pt is finite
                // program will crash when we conduct nearestKSearch() on a infinite point
                if (!pcl::isFinite(candidate_pt_pcl))
                {
                    continue;
                }

                CorridorPtr candidiate_corridor = std::make_shared<CorridorSample>();
                candidiate_corridor->geometry = GenerateOneSphere(candidate_pt);

                // skip the sample if it is too small
                // we assume the smallest_gap is the same as the drone_wheelbase_
                double smallest_gap = drone_wheelbase_;
                if (candidiate_corridor->geometry.second < smallest_gap || candidiate_corridor->geometry.second < 0)
                {
                    continue;
                }

                double candidate_radius = candidiate_corridor->geometry.second;
                double distance_between_center_and_guide_point = (candidate_pt - sample_origin).norm();
                if (distance_between_center_and_guide_point > candidate_radius)
                {
                    continue; // new sphere does not contain the guide point, so it is not useful
                }

                // if (sample_in_clutter)
                // {
                //     std::cout << "clutter sample radius is " << candidiate_corridor->geometry.second << std::endl;
                // }
                // std::cout << "candidiate_corridor " << candidiate_corridor->geometry.first.transpose() << " radius " << candidiate_corridor->geometry.second << std::endl;
                candidiate_corridor->updateScore(input_corridor, weighting);

                if (candidiate_corridor->score == 0)
                {
                    continue; // do not add candidate_corrior to pool if there is no overlap
                }
                else
                {
                    corridor_pool_.push(candidiate_corridor);
                }
            }
            if (corridor_pool_.size() == 0)
            {
                // std::cout << " no candidate found " << std::endl;
                return input_corridor;
            }

            // auto ret = corridor_pool_.top();
            // return ret->geometry;
            return corridor_pool_.top()->geometry;
        }
    }
}

// about return a vector efficiently
// https://stackoverflow.com/questions/28760475/how-to-return-a-class-member-vector-in-c11
const std::vector<Corridor> CorridorGenerator::getCorridor() const
{
    return flight_corridor_;
}

const std::vector<Eigen::Vector3d> CorridorGenerator::getWaypointList() const
{
    return waypoint_list_; // waypoints in sequence from start to end
}

const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> CorridorGenerator::getSampleDirection() const
{
    return sample_direction_;
}

const std::vector<Eigen::Vector3d> CorridorGenerator::getSamplePoint() const
{
    return sample_points_;
}

Eigen::Vector3d CorridorGenerator::getMidPointBetweenCorridors(const Corridor &previous_corridor, const Corridor &current_corridor)
{
    auto [pre_center, pre_radius] = previous_corridor;
    auto [curr_center, curr_radius] = current_corridor;
    // std::cout << "pre_center " << pre_center.transpose() << " pre_radius " << pre_radius << std::endl;
    // std::cout << "curr_center " << curr_center.transpose() << " curr_radius " << curr_radius << std::endl;
    Eigen::Vector3d previous2current;
    previous2current = curr_center - pre_center;
    double distance_between = previous2current.norm();
    previous2current = previous2current / distance_between;
    double distance_to_waypt = pre_radius - (pre_radius + curr_radius - distance_between) * 0.5; // *0.5 to replace division by 2
    Eigen::Vector3d waypt = previous2current * distance_to_waypt + pre_center;
    // std::cout << "waypt " << waypt.transpose() << std::endl;
    return waypt;
}

bool CorridorGenerator::corridorTooClose(const Corridor &corridor_for_check)
{
    if (flight_corridor_.empty())
    {
        return false;
    }

    if (bash_through_)
    {
        bash_through_ = false;
        return false;
    }

    double closest_distance = 100; // 100 as an arbitrarily large value

    auto [query_center, query_radius] = corridor_for_check;

    for (auto corridor : flight_corridor_)
    {
        auto [curr_center, curr_radius] = corridor;

        double distance = (curr_center - query_center).norm();

        if (distance < closest_distance)
        {
            closest_distance = distance;
        }
    }

    if (closest_distance < closeness_threshold_)
    {
        return true;
    }

    else
    {
        return false;
    }
}