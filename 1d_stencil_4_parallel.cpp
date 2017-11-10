//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the fourth in a series of examples demonstrating the development of
// a fully distributed solver for a simple 1D heat distribution problem.
//
// This example builds on example three. It futurizes the code from that
// example. Compared to example two this code runs much more efficiently. It
// allows for changing the amount of work executed in one HPX thread which
// enables tuning the performance for the optimal grain size of the
// computation. This example is still fully local but demonstrates nice
// scalability on SMP machines.

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>

#include <boost/iterator/counting_iterator.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "print_time_results.hpp"

///////////////////////////////////////////////////////////////////////////////
bool header = true; // print csv heading
double k = 0.5;     // heat transfer coefficient
double dt = 1.;     // time step
double dx = 1.;     // grid spacing
struct stepper_seq
{
    // Our partition type
    typedef double partition;

    // Our data for one time step
    typedef std::vector<partition> space;

    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k*dt / (dx*dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'nx' data points for 'nt' time steps
    space do_work(std::size_t nx, std::size_t nt)
    {
        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s : U)
            s.resize(nx);

        // Initial conditions: f(0, i) = i
        for (std::size_t i = 0; i != nx; ++i)
            U[0][i] = double(i);

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            next[0] = heat(current[nx - 1], current[0], current[1]);

            for (std::size_t i = 1; i != nx - 1; ++i)
                next[i] = heat(current[i - 1], current[i], current[i + 1]);

            next[nx - 1] = heat(current[nx - 2], current[nx - 1], current[0]);
        }

        // Return the solution at time-step 'nt'.
        return U[nt % 2];
    }
};
inline std::size_t idx(std::size_t i, std::size_t size)
{
    return (std::int64_t(i) < 0) ? (i + size) % size : i % size;
}

///////////////////////////////////////////////////////////////////////////////
// Our partition data type
struct partition_data
{
    partition_data(std::size_t size)
        : data_(new double[size]), size_(size)
    {}

    partition_data(std::size_t size, double initial_value)
        : data_(new double[size]), size_(size)
    {
        double base_value = double(initial_value * size);
        for (std::size_t i = 0; i != size; ++i)
            data_[i] = base_value + double(i);
    }

    double& operator[](std::size_t idx) { return data_[idx]; }
    double operator[](std::size_t idx) const { return data_[idx]; }

    std::size_t size() const { return size_; }

private:
    boost::shared_array<double> data_;
    std::size_t size_;
};

std::ostream& operator<<(std::ostream& os, partition_data const& c)
{
    os << "{";
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i != 0)
            os << ", ";
        os << c[i];
    }
    os << "}";
    return os;
}

///////////////////////////////////////////////////////////////////////////////
struct stepper
{
    // Our data for one time step
    typedef hpx::shared_future<partition_data> partition;
    typedef std::vector<partition> space;

    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k*dt / (dx*dx)) * (left - 2 * middle + right);
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition_data heat_part(partition_data const& left,
        partition_data const& middle, partition_data const& right)
    {
        std::size_t size = middle.size();
        partition_data next(size);

        typedef boost::counting_iterator<std::size_t> iterator;

        next[0] = heat(left[size - 1], middle[0], middle[1]);

        using namespace hpx::parallel;
        for_each(execution::par, iterator(1), iterator(size - 1),
            [&next, &middle](std::size_t i)
        {
            next[i] = heat(middle[i - 1], middle[i], middle[i + 1]);
        });

        next[size - 1] = heat(middle[size - 2], middle[size - 1], right[0]);

        return next;
    }

    // do all the work on 'np' partitions, 'nx' data points each, for 'nt'
    // time steps
    hpx::future<space> do_work(std::size_t np, std::size_t nx, std::size_t nt)
    {
        using hpx::util::unwrapping;
        using hpx::dataflow;
        using hpx::parallel::for_each;
        using hpx::parallel::execution::par;

        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s : U)
            s.resize(np);

        // Initial conditions: f(0, i) = i
        for (std::size_t i = 0; i != np; ++i)
            U[0][i] = hpx::make_ready_future(partition_data(nx, double(i)));

        auto Op = unwrapping(&stepper::heat_part);

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            typedef boost::counting_iterator<std::size_t> iterator;

            for_each(par, iterator(0), iterator(np),
                [&next, &current, np, &Op](std::size_t i)
            {
                next[i] = dataflow(
                    hpx::launch::async, Op,
                    current[idx(i - 1, np)], current[i], current[idx(i + 1, np)]
                );
            });
        }

        // Return the solution at time-step 'nt'.
        return hpx::when_all(U[nt % 2]);
    }
};

double stencil_1(uint64_t nx, uint64_t nt)
{

    stepper_seq step_seq;

    // Measure execution time.
    std::uint64_t t = hpx::util::high_resolution_clock::now();

    // Execute nt time steps on nx grid points.
    stepper_seq::space solution = step_seq.do_work(nx, nt);



    std::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    std::uint64_t const os_thread_count = hpx::get_os_thread_count();
    print_time_results(os_thread_count, elapsed, nx, nt, header);

    return double(elapsed / 1.0e9);
}

double stencil_4(uint64_t nx, uint64_t nt, uint64_t np)
{
    stepper step;

    // Measure execution time.
    std::uint64_t t = hpx::util::high_resolution_clock::now();

    // Execute nt time steps on nx grid points and print the final solution.
    hpx::future<stepper::space> result = step.do_work(np, nx, nt);

    stepper::space solution = result.get();
    hpx::wait_all(solution);

    std::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    // Print the final solution   

    std::uint64_t const os_thread_count = hpx::get_os_thread_count();
    print_time_results(os_thread_count, elapsed, nx, np, nt, header);

    return double(elapsed / 1.0e9);
}
void plot_vector(std::vector<std::vector<double>> v, std::string iname, std::string marker, std::string title, std::string xlabel, std::string ylabel, std::vector<std::string> legends)
{
    // Store the data in a text file
    const char *fname = "C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\data.txt";
    std::ofstream o(fname);
    for (int n = 0; n<v.size(); n++) {
        const std::vector<double>& vv = v[n];

        //o << std::endl;
        for (int i = 0; i<vv.size(); i++) {
            if (i > 0) o << ' ';
            o << vv[i];
        }
        o << std::endl;
    }
    o.close();

    // Create a python script to run matplotlib
    std::ostringstream cmd;
    cmd << "import matplotlib\n";
    cmd << "matplotlib.use('Agg')\n";
    cmd << "import numpy as np\n";
    cmd << "import matplotlib.pyplot as plt\n";
    cmd << "f = np.genfromtxt('" << fname << "')\n";
    cmd << "f = f.reshape(f.shape[0],-1)\n";
    cmd << "plt.figure()\n";
    cmd << "plt.title('" + title + "')\n";
    cmd << "plt.xlabel('" + xlabel + "')\n";
    cmd << "plt.ylabel('" + ylabel + "')\n";
    cmd << "for n in range(0,f.shape[0],2):\n";
    cmd << "  plt.plot(f[n,:],f[n+1,:],marker='" + marker + "')\n";
    cmd << "plt.legend(['Parallel','Sequential'],)\n";
    cmd << "plt.savefig('C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\" << iname << ".png')\n";
    cmd << "exit(0)\n";
    std::ofstream o2("C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\p.py");
    o2 << cmd.str();
    o2.close();
    system("c:\\users\\shahrzadshirzad\\anaconda3\\python.exe C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\p.py");

}
void plot_vector_1(std::vector<std::vector<double>> v, std::string iname, std::string marker, std::string title, std::string xlabel, std::string ylabel, std::vector<std::string> legends)
{
    // Store the data in a text file
    const char *fname = "C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\data.txt";
    std::ofstream o(fname);
    for (int n = 0; n<v.size(); n++) {
        const std::vector<double>& vv = v[n];

        //o << std::endl;
        for (int i = 0; i<vv.size(); i++) {
            if (i > 0) o << ' ';
            o << vv[i];
        }
        o << std::endl;
    }
    o.close();

    // Create a python script to run matplotlib
    std::ostringstream cmd;
    cmd << "import matplotlib\n";
    cmd << "matplotlib.use('Agg')\n";
    cmd << "import numpy as np\n";
    cmd << "import matplotlib.pyplot as plt\n";
    cmd << "f = np.genfromtxt('" << fname << "')\n";
    cmd << "f = f.reshape(f.shape[0],-1)\n";
    cmd << "plt.figure()\n";
    cmd << "plt.title('" + title + "')\n";
    cmd << "plt.xlabel('" + xlabel + "')\n";
    cmd << "plt.ylabel('" + ylabel + "')\n";
    cmd << "for n in range(0,f.shape[0],2):\n";
    cmd << "  plt.plot(f[n,:],f[n+1,:],marker='" + marker + "')\n";
    cmd << "plt.xscale('log')\n";
    cmd << "plt.legend(['Parallel','Sequential'],)\n";
    cmd << "plt.savefig('C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\" << iname << ".png')\n";
    cmd << "exit(0)\n";
    std::ofstream o2("C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\p.py");
    o2 << cmd.str();
    o2.close();
    system("c:\\users\\shahrzadshirzad\\anaconda3\\python.exe C:\\\\Users\\\\ShahrzadShirzad\\\\desktop\\\\Stencil\\\\p.py");

}
///////////////////////////////////////////////////////////////////////////////

int hpx_main(boost::program_options::variables_map& vm)
{
    using namespace hpx::parallel;
    uint64_t nx_total = 1e8;
    uint64_t n = log2(nx_total) / 2;
    uint64_t nt = 50;
    std::vector<std::vector<double>> plots;

    ////////////////////////////////////////////////////////////////////////////////
    double elapsed;
    double elapsed_1d;
    std::vector<double> elapsed_all;

    std::cout << "stencil_1 Results:" << std::endl;
    elapsed_1d = stencil_1(nx_total, nt);

    std::vector<double> num_partitions(n);
    int m = 1;
    num_partitions[0] = 1;
    std::generate(num_partitions.begin() + 1, num_partitions.end(), [&]() {return m = m * 2; });

    std::cout << "stencil_4 Results:" << std::endl;

    for (uint64_t p : num_partitions)
    {
        std::cout << p << std::endl;
        elapsed = stencil_4(nx_total / p, nt, p);
        std::cout << elapsed << std::endl;
        elapsed_all.push_back(elapsed);
    }
    std::transform(num_partitions.begin(), num_partitions.end(), num_partitions.begin(), [&](double i) {return  nx_total / i; });
    plots.push_back(num_partitions);
    plots.push_back(elapsed_all);
    std::vector<double> elapsed_1d_all(num_partitions);
    std::generate(elapsed_1d_all.begin(), elapsed_1d_all.end(), [&]() {return nx_total; });
    plots.push_back(elapsed_1d_all);
    std::generate(elapsed_1d_all.begin(), elapsed_1d_all.end(), [&]() {return elapsed_1d; });
    plots.push_back(elapsed_1d_all);
    plot_vector_1(plots, "test", "*", "", "Partition size", "Elapsed time(sec)", std::vector<std::string>{"1", "2"});
    ///////////////////////////////////////////////////////////////////////////////////////////


    //uint64_t step = (nx_total - 10e2) / n;
    //double elapsed;
    //double elapsed_1d;
    //std::vector<double> elapsed_all;


    //std::vector<double> num_partitions(9 * (log10(nx_total) - 2));
    //int k = 20;
    //int f = 0;
    //k = 0;
    //int m = 0;
    //for (int i = 0; i<10 * (log10(nx_total) - 2) - 1; i++)
    //{
    //    f++;

    //    if (f % 10 == 0)
    //        k++;
    //    else
    //    {
    //        num_partitions[m] = (f % 10)*pow(10, k);
    //        m++;
    //    }
    //}

    //std::cout << "stencil_4 Results:" << std::endl;

    //for (uint64_t p : num_partitions)
    //{
    //    std::cout << p << std::endl;
    //    elapsed = stencil_4(nx_total / p, nt, p);
    //    std::cout << elapsed << std::endl;
    //    elapsed_all.push_back(elapsed);
    //}
    //std::transform(num_partitions.begin(), num_partitions.end(), num_partitions.begin(), [&](double i) {return  nx_total / i; });
    //plots.push_back(num_partitions);
    //plots.push_back(elapsed_all);
    //plot_vector_1(plots, "test", "*", "", "Partition size", "Elapsed time(sec)", std::vector<std::string>{"1", "2"});

    //plot_vector_1(plots, "test", "*", "", "Partition size", "Elapsed time(sec)", std::vector<std::string>{"1", "2"});
    ////////////////////////////////////////////////////////////////////////////////////////
    /* uint64_t nx_total = 1e3;
    uint64_t n = log2(nx_total) / 2;
    uint64_t nt = 50;
    std::vector<std::vector<double>> plots;

    uint64_t step = (nx_total - 10e2) / n;
    double elapsed;
    double elapsed_1d;
    std::vector<double> elapsed_all;


    std::vector<double> points_per_partition(9 * (log10(nx_total) - 2));
    int k = 20;
    int f = 0;
    k = 2;
    int m = 0;
    for (int i = 0; i<10 * (log10(nx_total) - 2) - 1; i++)
    {
    f++;

    if (f % 10 == 0)
    k++;
    else
    {
    points_per_partition[m] = (f % 10)*pow(10, k);
    m++;
    }
    }

    std::cout << "stencil_4 Results:" << std::endl;

    for (uint64_t p : points_per_partition)
    {
    std::cout << p << std::endl;
    elapsed = stencil_4(p, nt, nx_total / p);
    std::cout << elapsed << std::endl;
    elapsed_all.push_back(elapsed);
    }
    plots.push_back(points_per_partition);
    plots.push_back(elapsed_all);
    plot_vector_1(plots, "test", "*", "", "Partition size", "Elapsed time(sec)", std::vector<std::string>{"1", "2"});*/
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //const char *fname = "C:/users/shahrzadshirzad/desktop/data.txt";
    //std::ofstream o(fname);
    ////std::uint64_t np = vm["np"].as<std::uint64_t>();   // Number of partitions.
    //std::uint64_t nx = vm["nx"].as<std::uint64_t>();   // Number of grid points.
    //std::uint64_t nt = vm["nt"].as<std::uint64_t>();   // Number of steps.
    //stepper_seq step_seq;

    //// Measure execution time.
    //std::uint64_t t = hpx::util::high_resolution_clock::now();

    //// Execute nt time steps on nx grid points.
    //stepper_seq::space solution = step_seq.do_work(nx, nt);


    //std::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;
    //
    //std::uint64_t const os_thread_count = hpx::get_os_thread_count();
    //o << "Number of threads: "<< os_thread_count<<std::endl <<"Example 1:" << std::endl << double(elapsed / 1.0e9) << std::endl << "Example 4:" << std::endl;

    //print_time_results(os_thread_count, elapsed, nx, nt, header);

    //for each (std::uint64_t np in std::vector<uint64_t>{ 1, 2, 4, 5, 10 })
    //{
    //     

    //    if (vm.count("no-header"))
    //        header = false;

    //    // Create the stepper object
    //    stepper step;

    //    // Measure execution time.
    //    t = hpx::util::high_resolution_clock::now();

    //    // Execute nt time steps on nx grid points and print the final solution.
    //    hpx::future<stepper::space> result = step.do_work(np, nx/np, nt);

    //    stepper::space solution = result.get();
    //    hpx::wait_all(solution);

    //    elapsed = hpx::util::high_resolution_clock::now() - t;

    //    // Print the final solution
    //    if (vm.count("result"))
    //    {
    //        for (std::size_t i = 0; i != np; ++i)
    //            std::cout << "U[" << i << "] = " << solution[i].get() << std::endl;
    //    }

    //    std::uint64_t const os_thread_count = hpx::get_os_thread_count();
    //    print_time_results(os_thread_count, elapsed, nx, np, nt, header);

    //    o << double(elapsed / 1.0e9)<<" ";
    //}
    //o.close();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    //std::vector<std::string> const cfg = {
    //    "hpx.os_threads=32"
    //};

    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("results,r", "print generated results (default: false)")
        ("nx", value<std::uint64_t>()->default_value(1000),
            "Local x dimension (of each partition)")
            ("nt", value<std::uint64_t>()->default_value(45),
                "Number of time steps")
                ("np", value<std::uint64_t>()->default_value(10),
                    "Number of partitions")
                    ("k", value<double>(&k)->default_value(0.5),
                        "Heat transfer coefficient (default: 0.5)")
                        ("dt", value<double>(&dt)->default_value(1.0),
                            "Timestep unit (default: 1.0[s])")
                            ("dx", value<double>(&dx)->default_value(1.0),
                                "Local x dimension")
                                ("no-header", "do not print out the csv header row")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
