#include <iostream>
#include <thread>
#include <atomic>
#include <omp.h>
using namespace std;

#define n 10000000000

atomic<unsigned> thread_num {thread::hardware_concurrency()};

void set_num_threads(int T)
{
    thread_num = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads()
{
    return thread_num;
}

typedef double (*f_t) (double);

double f(double x) {
    return x * x;
}

double w(double x) {
    return 2 * x;
}

double integrate_seq(double x1, double x2, f_t f, f_t w) {
    double res = 0.0;
    double dx = (x2 - x1) / n;
    size_t i;
    #pragma omp parallel for reduction(+: res)
    for (i = 0; i < n; ++i)
        res += f((dx * (double)i + x1)) * w((dx * (double)i + x1));
    return (1 / (x2 - x1)) * res * dx;
}

typedef struct experiment_result_t_ {
    double result;
    double time;
} experiment_result_t;

typedef double (*integrate_t)(double a, double b, f_t f, f_t w);

experiment_result_t run_experiment(integrate_t integrate) {
    experiment_result_t result;
    double t0 = omp_get_wtime();
    result.result = integrate(-1, 1, f, w);
    result.time = omp_get_wtime() - t0;
    return result;
}

void run_experiments(experiment_result_t* results, double (*I) (double, double, f_t, f_t)) {
    for (int T = 1; T <= thread::hardware_concurrency(); ++T) {
        set_num_threads(T);
        results[T - 1] = run_experiment(I);
    }
}

#include <iomanip>
void show_results_for(const char* name, const experiment_result_t* results) {
    int w = 10;
    cout << name << "\n";
    cout << std::setw(w) << "T" << "\t"
      << std::setw(w) << "Time" << "\t"
      << std::setw(w) << "Result" << "\t"
      << std::setw(w) << "Speedup\n";
    for (unsigned T = 1; T <= omp_get_num_procs(); T++)
        cout << std::setw(w) << T << "\t"
          << std::setw(w) << results[T - 1].time << "\t"
          << std::setw(w) << results[T - 1].result << "\t"
          << std::setw(w) << results[0].time / results[T - 1].time << "\n";
}

int main() {
    auto* results = (experiment_result_t*)malloc(get_num_threads() * sizeof(experiment_result_t));

    run_experiments(results, integrate_seq);
    show_results_for("integrate_seq", results);
}
