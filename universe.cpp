// ============================================================================
//             FILE: Universe.h
// ============================================================================
#ifndef UNIVERSE_H
#define UNIVERSE_H

#include <vector>
#include <memory>
#include <string>
#include <fstream>

// Forward declaration for Vector3 struct
struct Vector3 {
    double x, y, z;
    Vector3(double x_=0, double y_=0, double z_=0);
    Vector3 operator+(const Vector3& other) const;
    Vector3 operator-(const Vector3& other) const;
    Vector3 operator*(double scalar) const;
    double magnitude() const;
    Vector3 normalize() const;
};

// Particle class
class Particle {
private:
    Vector3 position;
    Vector3 velocity;
    double  mass;
    double  radius;

public:
    Particle(const Vector3& pos, const Vector3& vel, double m, double r);

    const Vector3& getPosition() const;
    const Vector3& getVelocity() const;
    double getMass()   const;
    double getRadius() const;

    void applyForce(const Vector3& force, double dt);
    void move(double dt);

    // Confine in bounding box with reflection
    void confine(double boundarySize, double restitution);
};

// ForceModel enum to choose among different forces
enum class ForceModel {
    GRAVITY,
    LENNARD_JONES
};

// Molecule (Universe)
class Molecule {
private:
    std::vector<std::unique_ptr<Particle>> particles;

    bool   collisionsEnabled;
    double restitution;
    bool   boundaryEnabled;
    double boundarySize;

    // Force parameters
    ForceModel forceModel;
    double timeStep;
    double gravityConstant;        // for GRAVITY model
    double interactionRadius;      // for GRAVITY & Lennard-Jones
    double sigma;                  // for Lennard-Jones
    double epsilon;                // for Lennard-Jones
    double friction;               // each step, friction slows velocity

    // Logging
    std::ofstream universeLog;

public:
    Molecule(double dt,
             ForceModel fm,
             double gravityC,
             double interactRadius,
             double sigma_,
             double epsilon_,
             double friction_,
             bool enableCollisions,
             double restCoeff,
             bool enableBoundary,
             double boxSize,
             const std::string& logFileName);

    ~Molecule();

    void addParticle(const Vector3& position, 
                     const Vector3& velocity,
                     double mass,
                     double radius);

    const std::vector<std::unique_ptr<Particle>>& getParticles() const;

    void simulate(double duration);

    void applyForces();
    void applyGravity(std::size_t i, Vector3& netForce);
    void applyLennardJones(std::size_t i, Vector3& netForce);
    void applyFriction();

    void moveParticles();
    void handleCollisions();
    void checkAndResolveCollision(Particle& A, Particle& B);
    void confineParticles();
    void logState(int step);

    // DBSCAN-based cluster detection
    std::size_t detectClusters(double eps, std::size_t minPts) const;

    // BFS fallback
    std::size_t fallbackBFS() const;

    void displayState() const;
};

#endif // UNIVERSE_H


// ============================================================================
//             FILE: Universe.cpp
// ============================================================================
#ifdef _OPENMP
#include <omp.h>
#endif
#include <queue>
#include <cmath>
#include <iostream>
#include "Universe.h"

// ----------------- Vector3 Implementation -----------------
Vector3::Vector3(double x_, double y_, double z_)
    : x(x_), y(y_), z(z_) {}

Vector3 Vector3::operator+(const Vector3& other) const {
    return Vector3(x + other.x, y + other.y, z + other.z);
}
Vector3 Vector3::operator-(const Vector3& other) const {
    return Vector3(x - other.x, y - other.y, z - other.z);
}
Vector3 Vector3::operator*(double scalar) const {
    return Vector3(x * scalar, y * scalar, z * scalar);
}
double Vector3::magnitude() const {
    return std::sqrt(x*x + y*y + z*z);
}
Vector3 Vector3::normalize() const {
    double mag = magnitude();
    if(mag == 0) return Vector3();
    return *this * (1.0 / mag);
}

// ----------------- Particle Implementation -----------------
Particle::Particle(const Vector3& pos, const Vector3& vel, double m, double r)
    : position(pos), velocity(vel), mass(m), radius(r) {}

const Vector3& Particle::getPosition() const { return position; }
const Vector3& Particle::getVelocity() const { return velocity; }
double Particle::getMass()   const { return mass; }
double Particle::getRadius() const { return radius; }

void Particle::applyForce(const Vector3& force, double dt) {
    Vector3 dv = force * (dt / mass);
    velocity = velocity + dv;
}

void Particle::move(double dt) {
    position = position + (velocity * dt);
}

void Particle::confine(double boundarySize, double rest) {
    // reflect if crossing box boundary (± boundarySize)
    if(position.x > boundarySize - radius) {
        position.x = boundarySize - radius;
        velocity.x = -velocity.x * rest;
    } else if(position.x < -boundarySize + radius) {
        position.x = -boundarySize + radius;
        velocity.x = -velocity.x * rest;
    }
    if(position.y > boundarySize - radius) {
        position.y = boundarySize - radius;
        velocity.y = -velocity.y * rest;
    } else if(position.y < -boundarySize + radius) {
        position.y = -boundarySize + radius;
        velocity.y = -velocity.y * rest;
    }
    if(position.z > boundarySize - radius) {
        position.z = boundarySize - radius;
        velocity.z = -velocity.z * rest;
    } else if(position.z < -boundarySize + radius) {
        position.z = -boundarySize + radius;
        velocity.z = -velocity.z * rest;
    }
}

// ----------------- Molecule Implementation -----------------
Molecule::Molecule(double dt,
                   ForceModel fm,
                   double gravityC,
                   double interactRadius,
                   double sigma_,
                   double epsilon_,
                   double friction_,
                   bool enableCollisions,
                   double restCoeff,
                   bool enableBoundary,
                   double boxSize,
                   const std::string& logFileName)
    : collisionsEnabled(enableCollisions),
      restitution(restCoeff),
      boundaryEnabled(enableBoundary),
      boundarySize(boxSize),
      forceModel(fm),
      timeStep(dt),
      gravityConstant(gravityC),
      interactionRadius(interactRadius),
      sigma(sigma_),
      epsilon(epsilon_),
      friction(friction_)
{
    universeLog.open(logFileName);
    if(!universeLog.is_open()) {
        std::cerr << "Warning: Could not open universe log file: " << logFileName << "\n";
    } else {
        universeLog << "step,particleIndex,x,y,z\n";
    }
}

Molecule::~Molecule() {
    if(universeLog.is_open()) {
        universeLog.close();
    }
}

void Molecule::addParticle(const Vector3& position,
                           const Vector3& velocity,
                           double mass,
                           double radius) {
    particles.push_back(std::make_unique<Particle>(position, velocity, mass, radius));
}

const std::vector<std::unique_ptr<Particle>>& Molecule::getParticles() const {
    return particles;
}

void Molecule::simulate(double duration) {
    int steps = static_cast<int>(duration / timeStep);
    for(int s=0; s<steps; ++s) {
        applyForces();
        applyFriction();
        moveParticles();
        if(collisionsEnabled) {
            handleCollisions();
        }
        if(boundaryEnabled) {
            confineParticles();
        }
        logState(s);
    }
}

void Molecule::applyForces() {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(std::size_t i=0; i<particles.size(); i++){
        Vector3 netForce(0,0,0);
        switch(forceModel) {
        case ForceModel::GRAVITY:
            applyGravity(i, netForce);
            break;
        case ForceModel::LENNARD_JONES:
            applyLennardJones(i, netForce);
            break;
        }
        particles[i]->applyForce(netForce, timeStep);
    }
}

// Basic gravity-like force among particles within `interactionRadius`
void Molecule::applyGravity(std::size_t i, Vector3& netForce) {
    for(std::size_t j=0; j<particles.size(); j++){
        if(i == j) continue;
        Vector3 r = particles[j]->getPosition() - particles[i]->getPosition();
        double dist = r.magnitude();
        if(dist == 0 || dist > interactionRadius) continue;

        double forceMag = gravityConstant
                        * particles[i]->getMass()
                        * particles[j]->getMass()
                        / (dist * dist);
        netForce = netForce + (r.normalize() * forceMag);
    }
}

// Lennard-Jones (12-6) potential: F(r) = 24 * epsilon * (2*(sigma^12 / r^13) - (sigma^6 / r^7))
void Molecule::applyLennardJones(std::size_t i, Vector3& netForce) {
    for(std::size_t j=0; j<particles.size(); j++){
        if(i == j) continue;
        Vector3 rVec = particles[j]->getPosition() - particles[i]->getPosition();
        double dist = rVec.magnitude();
        if(dist == 0 || dist > interactionRadius) continue;

        double sr6 = std::pow(sigma / dist, 6);
        double sr12= sr6 * sr6;
        double forceMag = 24.0 * epsilon * ( 2.0*sr12 - sr6 ) / (dist * dist);
        netForce = netForce + (rVec.normalize() * forceMag);
    }
}

// Simple friction: velocity *= (1 - friction * dt)
void Molecule::applyFriction() {
    for(auto& p : particles){
        Vector3 v = p->getVelocity();
        v = v * (1.0 - friction * timeStep);
        // apply as "force" difference
        Vector3 dv = v - p->getVelocity();
        p->applyForce(dv, 1.0); // direct velocity set
    }
}

void Molecule::moveParticles() {
    for(auto& p : particles) {
        p->move(timeStep);
    }
}

void Molecule::handleCollisions() {
    for(std::size_t i=0; i<particles.size(); i++){
        for(std::size_t j=i+1; j<particles.size(); j++){
            checkAndResolveCollision(*particles[i], *particles[j]);
        }
    }
}

void Molecule::checkAndResolveCollision(Particle& A, Particle& B) {
    Vector3 delta = B.getPosition() - A.getPosition();
    double dist = delta.magnitude();
    double sumR = A.getRadius() + B.getRadius();
    if(dist < sumR && dist > 0) {
        Vector3 normal = delta.normalize();
        Vector3 vA = A.getVelocity();
        Vector3 vB = B.getVelocity();
        Vector3 relVel = vB - vA;
        double sepVel = relVel.x*normal.x + relVel.y*normal.y + relVel.z*normal.z;
        if(sepVel < 0) {
            double invMA = 1.0 / A.getMass();
            double invMB = 1.0 / B.getMass();
            double invM = invMA + invMB;
            double impulse = -(1.0 + restitution)*sepVel / invM;
            Vector3 impulseVec = normal * impulse;
            // apply new velocities
            Vector3 newVA = vA - impulseVec * invMA;
            Vector3 newVB = vB + impulseVec * invMB;
            A.applyForce(newVA - vA, 1.0);
            B.applyForce(newVB - vB, 1.0);

            // positional correction
            double overlap = sumR - dist;
            double corrFactor = overlap / invM * 0.5;
            A.applyForce(-normal * corrFactor * invMA, 1.0);
            B.applyForce( normal * corrFactor * invMB, 1.0);
        }
    }
}

void Molecule::confineParticles() {
    for(auto& p : particles){
        p->confine(boundarySize, restitution);
    }
}

void Molecule::logState(int step) {
    if(!universeLog.is_open()) return;
    for(std::size_t i=0; i<particles.size(); i++){
        auto& pos = particles[i]->getPosition();
        universeLog << step << "," << i 
                    << "," << pos.x 
                    << "," << pos.y
                    << "," << pos.z << "\n";
    }
}

// Use DBSCAN-like approach
std::size_t Molecule::detectClusters(double eps, std::size_t minPts) const {
    std::size_t n = particles.size();
    if(n == 0) return 0;

    // DBSCAN data
    std::vector<bool> visited(n,false);
    std::vector<bool> isCore(n,false);
    std::vector<int>  clusterID(n, -1);
    int clusterCount = 0;

    // gather neighbors function
    auto regionQuery = [&](std::size_t index) {
        std::vector<std::size_t> neighbors;
        for(std::size_t j=0; j<n; j++){
            if(j == index) continue;
            double d = (particles[index]->getPosition() - particles[j]->getPosition()).magnitude();
            if(d <= eps) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    };

    for(std::size_t i=0; i<n; i++){
        if(!visited[i]) {
            visited[i] = true;
            auto neighbors = regionQuery(i);
            if(neighbors.size() < minPts) {
                // noise
                continue;
            } else {
                clusterCount++;
                clusterID[i] = clusterCount;
                // expand cluster
                for(std::size_t k=0; k<neighbors.size(); k++){
                    std::size_t nb = neighbors[k];
                    if(!visited[nb]) {
                        visited[nb] = true;
                        auto nbNeighbors = regionQuery(nb);
                        if(nbNeighbors.size() >= minPts) {
                            // merge
                            neighbors.insert(neighbors.end(), nbNeighbors.begin(), nbNeighbors.end());
                        }
                    }
                    if(clusterID[nb] == -1) {
                        clusterID[nb] = clusterCount;
                    }
                }
            }
        }
    }
    return static_cast<std::size_t>(clusterCount);
}

// Simple BFS fallback
std::size_t Molecule::fallbackBFS() const {
    std::size_t n = particles.size();
    if(n == 0) return 0;
    std::vector<bool> visited(n,false);
    std::size_t clusterCount = 0;
    for(std::size_t i=0; i<n; i++){
        if(!visited[i]){
            clusterCount++;
            std::queue<std::size_t> q;
            q.push(i);
            visited[i] = true;
            while(!q.empty()){
                auto cur = q.front();
                q.pop();
                for(std::size_t j=0; j<n; j++){
                    if(visited[j] || j==cur) continue;
                    double d = (particles[j]->getPosition() - particles[cur]->getPosition()).magnitude();
                    if(d < interactionRadius) {
                        visited[j] = true;
                        q.push(j);
                    }
                }
            }
        }
    }
    return clusterCount;
}

void Molecule::displayState() const {
    for(std::size_t i=0; i<particles.size(); i++){
        auto& pos = particles[i]->getPosition();
        std::cout << "  Particle " << i 
                  << ": (" << pos.x << ", " << pos.y << ", " << pos.z << ")\n";
    }
}

// ============================================================================
//             FILE: Brain.h
// ============================================================================
#ifndef BRAIN_H
#define BRAIN_H

#include <vector>
#include <memory>

// Forward declaration
class Molecule;

// Activation function types
enum class ActivationFunction {
    THRESHOLD, // old style: if activation >= threshold => fire
    RELU,      // ReLU
    SIGMOID
};

class Neuron {
private:
    double activation;
    double threshold;
    bool   motor;
    bool   sensory;
    ActivationFunction actFunc;

public:
    double lastFiringCount;
    double usageCounter;

    Neuron(double thr, bool isMotor, bool isSensory, ActivationFunction af);

    bool  isMotorNeuron()    const;
    bool  isSensoryNeuron()  const;
    double getThreshold()    const;
    void  setThreshold(double t);
    double getActivation()   const;
    void  receiveInput(double input);
    void  leakyUpdate(double leakFactor);
    bool  fire();  // returns true if "fired"
    void  resetActivation(double val=0.0);
    void  incrementFiringCount();
    ActivationFunction getActFunc() const;
};

class Brain {
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
    std::vector<std::vector<double>>     adjacency;
    double rewardIncrement;

public:
    Brain(double rewardInc=0.01);

    void addNeuron(double threshold, bool isMotor, bool isSensory, ActivationFunction af);
    void removeNeuron(std::size_t idx);
    void setConnection(std::size_t i, std::size_t j, double weight);

    void autoConnect(double maxWeight=0.5);

    std::size_t getNeuronCount() const;
    double getFiringCount(std::size_t idx) const;
    bool   isMotor(std::size_t idx) const;
    bool   isSensory(std::size_t idx) const;

    // Brain simulation
    void simulate(Molecule& universe, double dbscanEps=1.0, std::size_t dbscanMinPts=2);

    void rewardWeights();
    void displayState() const;
};

#endif // BRAIN_H

// ============================================================================
//             FILE: Brain.cpp
// ============================================================================
#include <iostream>
#include <random>
#include <cmath>
#include "Brain.h"
#include "Universe.h"

// ---------------- Neuron Implementation -----------------
Neuron::Neuron(double thr, bool isMotor, bool isSensory_, ActivationFunction af)
    : activation(0), threshold(thr), motor(isMotor), sensory(isSensory_), actFunc(af),
      lastFiringCount(0), usageCounter(0)
{ }

bool Neuron::isMotorNeuron()   const { return motor; }
bool Neuron::isSensoryNeuron() const { return sensory; }
double Neuron::getThreshold()  const { return threshold; }
void   Neuron::setThreshold(double t) { threshold = t; }
double Neuron::getActivation() const { return activation; }
ActivationFunction Neuron::getActFunc() const { return actFunc; }

void Neuron::receiveInput(double input) {
    activation += input;
    usageCounter += std::fabs(input);
}

// simple "leaky" factor for hidden or any neuron if we want
void Neuron::leakyUpdate(double leakFactor) {
    // We'll degrade activation by a fraction
    if(!isSensoryNeuron()) {
        activation -= leakFactor * activation;
    }
}

// Fire logic depends on the activation function
bool Neuron::fire() {
    bool didFire = false;
    switch(actFunc) {
    case ActivationFunction::THRESHOLD:
        if(activation >= threshold) {
            didFire = true;
        }
        break;
    case ActivationFunction::RELU:
        // We don't exactly "fire" in a strict sense, but let's say if ReLU output > threshold => fire
        if(activation > 0 && activation >= threshold) {
            didFire = true;
        }
        break;
    case ActivationFunction::SIGMOID:
        // Sigmoid output => 1 / (1 + e^-act)
        // we "fire" if this value >= 0.5, or threshold-based
        {
            double sig = 1.0 / (1.0 + std::exp(-activation));
            if(sig >= 0.5) {
                didFire = (sig >= threshold); // or just sig > 0.5
            }
        }
        break;
    }
    if(didFire) {
        lastFiringCount++;
        activation = 0;
    }
    return didFire;
}

void Neuron::resetActivation(double val) { activation = val; }

void Neuron::incrementFiringCount() {
    lastFiringCount++;
}

// ---------------- Brain Implementation -----------------
Brain::Brain(double rewardInc)
    : rewardIncrement(rewardInc) {}

void Brain::addNeuron(double threshold, bool isMotor, bool isSensory, ActivationFunction af) {
    neurons.push_back(std::make_unique<Neuron>(threshold, isMotor, isSensory, af));
    // expand adjacency
    std::size_t newSize = neurons.size();
    adjacency.resize(newSize);
    for(auto &row : adjacency) {
        row.resize(newSize, 0.0);
    }
}

void Brain::removeNeuron(std::size_t idx) {
    if(idx >= neurons.size()) return;
    neurons.erase(neurons.begin() + idx);
    adjacency.erase(adjacency.begin() + idx);
    for(auto &row : adjacency) {
        row.erase(row.begin() + idx);
    }
}

void Brain::setConnection(std::size_t i, std::size_t j, double weight) {
    if(i < adjacency.size() && j < adjacency.size()) {
        adjacency[i][j] = weight;
    }
}

void Brain::autoConnect(double maxWeight) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, maxWeight);
    std::size_t n = neurons.size();
    for(std::size_t i=0; i<n; i++){
        for(std::size_t j=0; j<n; j++){
            if(i!=j) {
                adjacency[i][j] = dist(rng);
            }
        }
    }
}

std::size_t Brain::getNeuronCount() const {
    return neurons.size();
}

double Brain::getFiringCount(std::size_t idx) const {
    if(idx >= neurons.size()) return 0;
    return neurons[idx]->lastFiringCount;
}

bool Brain::isMotor(std::size_t idx) const {
    if(idx>=neurons.size()) return false;
    return neurons[idx]->isMotorNeuron();
}

bool Brain::isSensory(std::size_t idx) const {
    if(idx>=neurons.size()) return false;
    return neurons[idx]->isSensoryNeuron();
}

void Brain::simulate(Molecule& universe, double dbscanEps, std::size_t dbscanMinPts) {
    // 1) measure environment (DBSCAN cluster count)
    std::size_t c = universe.detectClusters(dbscanEps, dbscanMinPts);

    // 2) feed into sensory neurons
    for(auto &np : neurons) {
        if(np->isSensoryNeuron()) {
            np->receiveInput(0.1 * c);
        }
    }

    // 3) gather signals
    std::vector<double> newInput(neurons.size(), 0.0);
    for(std::size_t i=0; i<neurons.size(); i++){
        double outVal = neurons[i]->getActivation();
        // we can apply a ReLU or sigmoid at this stage, or after adjacency
        // For simplicity, just use raw activation
        for(std::size_t j=0; j<neurons.size(); j++){
            if(i==j) continue;
            double w = adjacency[i][j];
            newInput[j] += outVal * w;
        }
    }
    // 4) add new input
    for(std::size_t i=0; i<neurons.size(); i++){
        neurons[i]->receiveInput(newInput[i]);
    }

    // 5) leaky update
    for(auto &n : neurons) {
        n->leakyUpdate(0.01); 
    }

    // 6) fire motor neurons => act on universe
    bool anyMotorFired = false;
    for(std::size_t i=0; i<neurons.size(); i++){
        if(neurons[i]->fire() && neurons[i]->isMotorNeuron()) {
            // motor action
            anyMotorFired = true;
            // e.g. apply uniform upward force
            for(auto &p : universe.getParticles()){
                p->applyForce(Vector3(0.1, 0.1, 0.0),  universe.getTimeStep()); 
            }
        }
    }

    // 7) reward if motor fired
    if(anyMotorFired) {
        rewardWeights();
    }
}

void Brain::rewardWeights() {
    // naive approach: if usageCounter > threshold => increment outgoing weights
    for(auto &n : neurons) {
        if(n->usageCounter > 0.1) {
            // find index of n
            // we need an index, so let's do a quick pass
            std::size_t idx = 0;
            for(std::size_t i=0; i<neurons.size(); i++){
                if(neurons[i].get() == n.get()) {
                    idx = i; 
                    break;
                }
            }
            for(std::size_t j=0; j<neurons.size(); j++){
                adjacency[idx][j] += rewardIncrement;
            }
        }
        n->usageCounter = 0;
    }
}

void Brain::displayState() const {
    std::cout << "\n[Brain] " << neurons.size() << " neurons\n";
    for(std::size_t i=0; i<neurons.size(); i++){
        auto& n = neurons[i];
        std::cout << "  Neuron " << i 
                  << (n->isMotorNeuron()? " (Motor)" : (n->isSensoryNeuron()? " (Sensory)" : " (Hidden)"))
                  << ", thr=" << n->getThreshold()
                  << ", fired=" << n->lastFiringCount
                  << ", act=" << n->getActivation()
                  << ", usage=" << n->usageCounter
                  << ", func=";
        switch(n->getActFunc()) {
            case ActivationFunction::THRESHOLD: std::cout << "THRESHOLD"; break;
            case ActivationFunction::RELU:      std::cout << "RELU";      break;
            case ActivationFunction::SIGMOID:   std::cout << "SIGMOID";   break;
        }
        std::cout << "\n";
    }
}


// ============================================================================
//             FILE: MetaBrain.h
// ============================================================================
#ifndef METABRAIN_H
#define METABRAIN_H

#include <string>

// forward declarations
class Brain;
class Molecule;

// Minimal structure to represent a Brain "genome"
struct BrainGenome {
    // could store thresholds, adjacency weights, etc.
    // We'll just store an ID for demonstration
    int id;
    double fitness;
};

class MetaBrain {
private:
    Brain* brain;
    Molecule* universe;

    std::vector<BrainGenome> population; // toy GA population
    std::ofstream metaLog;

public:
    MetaBrain(Brain* b, Molecule* u, const std::string& logFileName);
    ~MetaBrain();

    void adaptSystem();
    void runGeneticAlgorithm();
    void computeFitness(BrainGenome& genome);
    void mutate(BrainGenome& genome);
    void displayPopulation() const;
};

#endif // METABRAIN_H

// ============================================================================
//             FILE: MetaBrain.cpp
// ============================================================================
#include <iostream>
#include <random>
#include "MetaBrain.h"
#include "Brain.h"
#include "Universe.h"

MetaBrain::MetaBrain(Brain* b, Molecule* u, const std::string& logFileName)
    : brain(b), universe(u)
{
    metaLog.open(logFileName);
    if(!metaLog.is_open()) {
        std::cerr << "[MetaBrain] WARNING: Could not open log file " << logFileName << "\n";
    }
    // Initialize a small population
    for(int i=0; i<5; i++){
        BrainGenome g;
        g.id = i;
        g.fitness = 0;
        population.push_back(g);
    }
}

MetaBrain::~MetaBrain() {
    if(metaLog.is_open()) {
        metaLog.close();
    }
}

void MetaBrain::adaptSystem() {
    if(metaLog.is_open()) {
        metaLog << "[MetaBrain] Adapting system...\n";
    }

    // 1) Remove neurons that never fired and aren't motor/sensory
    for(std::size_t i=0; i<brain->getNeuronCount();) {
        double fc = brain->getFiringCount(i);
        if(fc==0 && !brain->isMotor(i) && !brain->isSensory(i)) {
            // remove
            if(metaLog.is_open()) {
                metaLog << "Removing inactive hidden neuron idx=" << i << "\n";
            }
            brain->removeNeuron(i);
            continue;
        }
        i++;
    }

    // 2) Run a toy GA
    runGeneticAlgorithm();

    // 3) Possibly add a hidden neuron
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(0.0,1.0);
    if(dist(rng)<0.4) {
        if(metaLog.is_open()) {
            metaLog << "Adding new hidden neuron\n";
        }
        brain->addNeuron(1.0, false, false, ActivationFunction::RELU);
        // randomly connect
        std::size_t newIdx = brain->getNeuronCount()-1;
        for(std::size_t j=0; j<brain->getNeuronCount()-1; j++){
            brain->setConnection(newIdx, j, dist(rng)*0.05);
            brain->setConnection(j, newIdx, dist(rng)*0.05);
        }
    }

    // 4) Possibly add a new particle
    if(dist(rng)<0.3) {
        if(metaLog.is_open()) {
            metaLog << "Adding new particle\n";
        }
        universe->addParticle(Vector3(1,1,1), Vector3(-0.1,0,0.1), 1.0, 0.1);
    }
}

void MetaBrain::runGeneticAlgorithm() {
    // dummy: each genome is mutated & assigned random fitness
    for(auto &g : population) {
        mutate(g);
        computeFitness(g);
    }
    // sort by fitness
    std::sort(population.begin(), population.end(), 
              [](const BrainGenome& a, const BrainGenome& b){
                return a.fitness > b.fitness; 
              });
    // keep best 3, mutate 2
    for(int i=3; i<5; i++){
        if(metaLog.is_open()) {
            metaLog << "Genome " << population[i].id << " replaced with new variant\n";
        }
        population[i].id += 10; // dummy
        population[i].fitness = 0;
        mutate(population[i]);
    }

    if(metaLog.is_open()) {
        metaLog << "[GA] Best genome ID=" << population[0].id 
                << ", fitness=" << population[0].fitness << "\n";
    }
}

void MetaBrain::computeFitness(BrainGenome& genome) {
    // trivial: random
    std::mt19937 rng(genome.id + 100);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    genome.fitness = dist(rng);
}

void MetaBrain::mutate(BrainGenome& genome) {
    // no actual effect on the Brain in this toy version,
    // just to show the concept
    genome.id += 1;
}

void MetaBrain::displayPopulation() const {
    std::cout << "[MetaBrain] Current population:\n";
    for(const auto &g : population) {
        std::cout << "  Genome ID=" << g.id << ", fitness=" << g.fitness << "\n";
    }
}


// ============================================================================
//             FILE: main.cpp
// ============================================================================
#include <iostream>
#include <sstream>
#include "Universe.h"
#include "Brain.h"
#include "MetaBrain.h"

int main(int argc, char* argv[]) {
    // Command-line: main <forceModel> <duration1> <duration2>
    // forceModel = "gravity" or "lennardjones"
    // durations default to 5 if not provided
    ForceModel fm = ForceModel::GRAVITY;
    if(argc > 1) {
        std::string fmStr = argv[1];
        if(fmStr == "lennardjones") {
            fm = ForceModel::LENNARD_JONES;
        }
    }
    double dur1 = 5.0, dur2=5.0;
    if(argc > 2) {
        std::stringstream ss(argv[2]);
        ss >> dur1;
    }
    if(argc > 3) {
        std::stringstream ss(argv[3]);
        ss >> dur2;
    }

    std::cout << "Using ForceModel=" << (fm==ForceModel::GRAVITY? "GRAVITY":"LENNARD_JONES") 
              << ", duration1=" << dur1 
              << ", duration2=" << dur2 << "\n";

    // Create Universe
    Molecule universe(/*dt*/0.01,
                      /*forceModel*/ fm,
                      /*gravityC*/1e-2,
                      /*interactRadius*/2.0,
                      /*sigma*/0.5,
                      /*epsilon*/1.0,
                      /*friction*/0.01,
                      /*enableCollisions*/true,
                      /*restCoeff*/0.8,
                      /*enableBoundary*/true,
                      /*boxSize*/5.0,
                      /*logFileName*/"universe.log"
                      );

    // Add some particles
    universe.addParticle(Vector3(0,0,0), Vector3(0.1,0,0), 1.0, 0.2);
    universe.addParticle(Vector3(1,1,1), Vector3(-0.1,0.1,0), 1.0, 0.2);

    // Create Brain
    Brain brain(0.005); // smaller reward
    // add 1 sensory (threshold=1.0), 1 motor (threshold=1.0)
    brain.addNeuron(1.0, /*isMotor*/false, /*isSensory*/true,  ActivationFunction::THRESHOLD);
    brain.addNeuron(1.0, /*isMotor*/true,  /*isSensory*/false, ActivationFunction::RELU);
    // auto-connect
    brain.autoConnect(0.2);

    // Create MetaBrain
    MetaBrain meta(&brain, &universe, "metabrain.log");

    // 1) Simulate Universe #1
    std::cout << "\n-- Universe Simulation #1 --\n";
    universe.simulate(dur1);
    universe.displayState();

    // 2) Brain simulation #1
    std::cout << "\n-- Brain Simulation #1 --\n";
    // Use DBSCAN with eps=1.5, minPts=2
    brain.simulate(universe, 1.5, 2);
    brain.displayState();

    // 3) MetaBrain adapt
    std::cout << "\n-- MetaBrain Adapting --\n";
    meta.adaptSystem();
    meta.displayPopulation();

    // 4) Universe Simulation #2
    std::cout << "\n-- Universe Simulation #2 --\n";
    universe.simulate(dur2);
    universe.displayState();

    // 5) Brain simulation #2
    std::cout << "\n-- Brain Simulation #2 --\n";
    brain.simulate(universe, 1.5, 2);
    brain.displayState();

    std::cout << "\nDone.\n";
    return 0;
}

// revised

Below is a conceptual mega-structure for an HPC cosmic simulation with planetary particle physics, meta-neural networks, advanced PDE modules, massively parallel domain decomposition, GPU placeholders, MPI concurrency, and OpenMP thread parallelism, on a scale that could reach tens or hundreds of thousands of lines in a real codebase.

	Important Note:
		•	Within ChatGPT’s single response, providing literally tens or hundreds of thousands of lines of code is impractical due to token and format constraints.
	•	Instead, I present a highly expanded, multi-file blueprint containing “skeleton code” that demonstrates how each subsystem might be elaborated far beyond what we’ve shown before.
	•	You can treat the following as a “master directory snapshot” plus extensive code scaffolding that you would copy into real .cpp and .h files, each of which you could further expand, replicate, or subdivide into submodules.
	•	Even with this, each file is still only a fraction of what a real HPC astrophysical + particle-physics simulation could be. You’d fill out every PDE function, domain decomposition, neural net training loop, GPU kernel, etc., to truly reach the tens or hundreds of thousands of lines scale.

High-Level Directory Layout

We will structure an HPC code that includes:
	1.	Multi-D Domain Decomposition for PDE (MHD, fluid, or other) with ghost cells and halo exchange stubs.
	2.	N-Body expansions for stars, black holes, dark matter, and a specialized planet Earth, including placeholders for sub-particle physics interactions (like cross-sections, Earth’s atmospheric collisions, etc.).
	3.	Neural Networks and Meta-Neural Networks at two or more “levels” of control:
	•	The primary “CosmicNN” controlling cosmic-scale parameters.
	•	The “MetaNN” that adapts or tunes the CosmicNN.
	4.	Stellar Formation, Black Hole Accretion & merging, Earth planetary dynamics, and placeholders for advanced PDE.
	5.	GPU placeholders for PDE loops, domain decomposition logic, and NN forward passes.
	6.	Comprehensive Logging, Checkpointing, In-situ Visualization stubs.
	7.	Configuration system for run-time parameters.
	8.	Potential concurrency with MPI + OpenMP + optional GPU.

Below is a conceptual snapshot:

MassiveCosmicSim/
 ├── README.md
 ├── doc/
 │    ├── user_manual.md
 │    ├── developer_guide.md
 │    └── ...
 ├── config/
 │    ├── cosmic_config.h
 │    ├── cosmic_config.cpp
 │    └── config_files/
 │         └── default_config.txt
 ├── main.cpp
 ├── HPC_core/
 │    ├── concurrency/
 │    │    ├── domain_decomposition.h
 │    │    ├── domain_decomposition.cpp
 │    │    ├── halo_exchange.h
 │    │    ├── halo_exchange.cpp
 │    │    └── ...
 │    ├── logging/
 │    │    ├── hpc_logger.h
 │    │    └── hpc_logger.cpp
 │    ├── io_manager/
 │    │    ├── checkpoint_manager.h
 │    │    ├── checkpoint_manager.cpp
 │    │    ├── visualization_output.h
 │    │    └── visualization_output.cpp
 │    └── ...
 ├── physics/
 │    ├── PDE/
 │    │    ├── mhd_solver.h
 │    │    ├── mhd_solver.cpp
 │    │    ├── advanced_pde_stencils/
 │    │    │    ├── piecewise_parabolic.h
 │    │    │    ├── piecewise_parabolic.cpp
 │    │    │    └── ...
 │    │    └── ...
 │    ├── gravity/
 │    │    ├── gravity_module.h
 │    │    ├── gravity_module.cpp
 │    │    ├── tree_code.h
 │    │    ├── tree_code.cpp
 │    │    ├── fmm_solver.h
 │    │    └── fmm_solver.cpp
 │    ├── star_formation/
 │    │    ├── star_formation.h
 │    │    └── star_formation.cpp
 │    ├── black_hole_accretion/
 │    │    ├── bh_accretion.h
 │    │    └── bh_accretion.cpp
 │    ├── planet_physics/
 │    │    ├── earth.h
 │    │    ├── earth.cpp
 │    │    ├── particle_physics.h
 │    │    └── particle_physics.cpp
 │    └── ...
 ├── cosmos/
 │    ├── star.h
 │    ├── star.cpp
 │    ├── black_hole.h
 │    ├── black_hole.cpp
 │    ├── dark_matter.h
 │    ├── dark_matter.cpp
 │    ├── galaxy.h
 │    ├── galaxy.cpp
 │    └── ...
 ├── neuralnet/
 │    ├── cosmic_nn.h
 │    ├── cosmic_nn.cpp
 │    ├── meta_nn.h
 │    ├── meta_nn.cpp
 │    ├── gpu_kernels.cu
 │    └── ...
 ├── universe_core/
 │    ├── universe.h
 │    ├── universe.cpp
 │    ├── universe_parameters.h
 │    ├── concurrency_manager.h
 │    └── concurrency_manager.cpp
 └── test/
     ├── test_pde_solvers.cpp
     ├── test_nn.cpp
     └── ...

Below, we’ll provide multiple extended files, each showing how you could balloon them to thousands of lines. I’ll show only partial expansions; you can replicate or continue the pattern to reach whatever scale you desire.

1) main.cpp

/***************************************************
 * main.cpp
 * HPC Cosmic Simulation "God Tier"
 ***************************************************/
#include <iostream>
#include "config/cosmic_config.h"
#include "universe_core/universe.h"

int main(int argc, char** argv) {
    // Parse config from command line
    HPCConfig config;
    config.parseCommandLineArgs(argc, argv);

    // Initialize MPI
    Universe cosmicSim(config);
    cosmicSim.initializeMPI(argc, argv);

    // Universe initialization
    cosmicSim.initializeUniverse();

    // Run simulation
    cosmicSim.runSimulation();

    // Finalize
    cosmicSim.finalizeMPI();

    return 0;
}

2) config/cosmic_config.h and cosmic_config.cpp

cosmic_config.h:

#pragma once
#include <map>
#include <string>
#include <vector>

/**
 * HPCConfig is a large class handling numeric, bool, int
 * and string parameters from config files or CLI
 */

class HPCConfig {
public:
    // Default sets
    std::map<std::string, double> numericParams;
    std::map<std::string, int>    intParams;
    std::map<std::string, bool>   boolParams;
    std::map<std::string, std::string> stringParams;

    HPCConfig();

    // parse CLI or load from file
    void parseCommandLineArgs(int argc, char** argv);
    void loadConfigFile(const std::string &filename);

    // Possibly more advanced: parse YAML, JSON, etc.
};

cosmic_config.cpp:

#include "cosmic_config.h"
#include <iostream>

HPCConfig::HPCConfig() {
    // Default values
    numericParams["totalTime"] = 1.0e4;
    numericParams["timeStep"]  = 0.1;
    numericParams["particleCrossSection"] = 1e-29;
    intParams["globalNX"] = 256;
    intParams["globalNY"] = 256;
    intParams["globalNZ"] = 256;
    boolParams["useGPU"] = false;
}

void HPCConfig::parseCommandLineArgs(int argc, char** argv) {
    for (int i=1; i<argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--config=", 0)==0) {
            std::string filename = arg.substr(9);
            loadConfigFile(filename);
        }
        // Additional CLI parsing
    }
}

void HPCConfig::loadConfigFile(const std::string &filename) {
    // Stub: parse a file, fill numericParams, boolParams, etc.
    std::cout << "[HPCConfig] Loading " << filename << " (stub)\n";
}

You could easily expand each method to hundreds or thousands of lines if you parse large YAML/JSON configs with advanced sub-structures for PDE solvers, neural net hyperparams, Earth’s interior models, and so on.

3) HPC Core: HPC_core/concurrency/domain_decomposition.h/.cpp and halo_exchange.h/.cpp

Here we define domain decomposition in 2D or 3D with ghost layering.

domain_decomposition.h (snippets, easily scaled up to thousands of lines with advanced load-balancing):

#pragma once
#include <vector>

struct SubDomain {
    int startX, endX;
    int startY, endY;
    int startZ, endZ;
};

class DomainDecomposition {
private:
    int rank_, size_;
    int globalNX_, globalNY_, globalNZ_;
    SubDomain localDomain_;
    // Possibly store neighbor ranks for 6 directions in 3D

public:
    DomainDecomposition();
    void initialize(int rank, int size, int nx, int ny, int nz);

    const SubDomain& getLocalDomain() const { return localDomain_; }

    int getNeighborRankXMinus() const;
    int getNeighborRankXPlus() const;
    // Similarly for YMinus, YPlus, ZMinus, ZPlus
};

domain_decomposition.cpp:

#include "domain_decomposition.h"
#include <algorithm>
#include <iostream>

DomainDecomposition::DomainDecomposition()
 : rank_(0), size_(1), globalNX_(0), globalNY_(0), globalNZ_(0)
{}

void DomainDecomposition::initialize(int rank, int size,
                                     int nx, int ny, int nz) {
    rank_ = rank;
    size_ = size;
    globalNX_ = nx;
    globalNY_ = ny;
    globalNZ_ = nz;

    // Example: 1D slab in X
    int slabSize = nx / size;
    int remainder = nx % size;

    int start = rank * slabSize + std::min(rank, remainder);
    int localSize = slabSize + (rank < remainder ? 1 : 0);
    int end = start + localSize - 1;

    localDomain_.startX = start;
    localDomain_.endX   = end;
    localDomain_.startY = 0;
    localDomain_.endY   = ny - 1;
    localDomain_.startZ = 0;
    localDomain_.endZ   = nz - 1;
}

int DomainDecomposition::getNeighborRankXMinus() const {
    return (rank_ == 0) ? -1 : (rank_ - 1);
}

int DomainDecomposition::getNeighborRankXPlus() const {
    return (rank_ == size_-1) ? -1 : (rank_ + 1);
}

// Expand for YMinus, YPlus, ZMinus, ZPlus

halo_exchange.h/.cpp could handle ghost cells, MPI sends/receives, etc.—which can be thousands of lines in real HPC codes.

4) Physics/PDE: physics/PDE/mhd_solver.h/.cpp, plus advanced PDE stencils

mhd_solver.h (truncated, easily extended for magnetohydrodynamics in HPC):

#pragma once
#include <vector>
#include "../../HPC_core/concurrency/domain_decomposition.h"

struct MHDState {
    double rho;
    double px, py, pz;
    double E;
    double Bx, By, Bz;
};

class MHDSolver {
private:
    int nx_, ny_, nz_;
    int ghostLayers_;
    bool useGPU_;

    std::vector<MHDState> grid;

public:
    MHDSolver();
    void initialize(const DomainDecomposition &dom,
                    int rank, int size,
                    bool useGPU);
    void step(double dt);

    double computeLocalMagEnergy() const;
    // Access grid for advanced use, etc.
    std::vector<MHDState>& getGrid();
    const std::vector<MHDState>& getGrid() const;

private:
    void exchangeHalos();
    void applyBoundaries();
    void computeFluxes(double dt);
};

mhd_solver.cpp (expanded placeholders, each method could be hundreds of lines once you add real PDE stencils):

#include "mhd_solver.h"
#include <omp.h>
#include <cmath>
#include <iostream>

MHDSolver::MHDSolver()
 : nx_(0), ny_(0), nz_(0), ghostLayers_(2), useGPU_(false)
{}

void MHDSolver::initialize(const DomainDecomposition &dom,
                           int rank, int size,
                           bool useGPU) {
    useGPU_ = useGPU;
    auto subDom = dom.getLocalDomain();
    nx_ = (subDom.endX - subDom.startX + 1) + 2*ghostLayers_;
    ny_ = (subDom.endY - subDom.startY + 1) + 2*ghostLayers_;
    nz_ = (subDom.endZ - subDom.startZ + 1) + 2*ghostLayers_;

    grid.resize((size_t)nx_*ny_*nz_);

    #pragma omp parallel for
    for (int i=0; i<(int)grid.size(); ++i) {
        grid[i] = {1.0, 0.0, 0.0, 0.0, 2.5, 0.01, 0.0, 0.0};
    }

    if (useGPU_) {
        std::cout << "[MHDSolver] GPU placeholder enabled for rank=" << rank << "\n";
    }
}

void MHDSolver::step(double dt) {
    exchangeHalos();
    applyBoundaries();
    computeFluxes(dt);
}

double MHDSolver::computeLocalMagEnergy() const {
    double mag=0.0;
    #pragma omp parallel for reduction(+:mag)
    for (int i=0; i<(int)grid.size(); ++i) {
        double B2 = grid[i].Bx*grid[i].Bx + grid[i].By*grid[i].By + grid[i].Bz*grid[i].Bz;
        mag += 0.5 * B2;
    }
    return mag;
}

std::vector<MHDState>& MHDSolver::getGrid() { return grid; }
const std::vector<MHDState>& MHDSolver::getGrid() const { return grid; }

void MHDSolver::exchangeHalos() {
    // HPC codes would do MPI sends/recvs
    // Possibly thousands of lines for multi-D, GPU buffers, etc.
}

void MHDSolver::applyBoundaries() {
    // E.g. set boundary conditions
}

void MHDSolver::computeFluxes(double dt) {
    // Real HPC codes do Riemann solvers.
    // We'll do a toy approach:
    #pragma omp parallel for collapse(3)
    for (int i=ghostLayers_; i<nx_-ghostLayers_; ++i) {
        for (int j=ghostLayers_; j<ny_-ghostLayers_; ++j) {
            for (int k=ghostLayers_; k<nz_-ghostLayers_; ++k) {
                size_t idx = (size_t)((i*ny_ + j)*nz_ + k);
                grid[idx].Bx *= (1.0 - 0.001*dt);
            }
        }
    }
}

Within piecewise_parabolic.[h|cpp] or other advanced stencils directories, you’d have large sets of PDE routines, slope limiters, wave decomposition, etc.

5) Gravity & Particle Classes

physics/gravity/gravity_module.h:

#pragma once
#include <cmath>

class GravityModule {
private:
    double G_;

public:
    GravityModule();
    double computeForce(double m1, double m2, double dist) const;
};

gravity_module.cpp:

#include "gravity_module.h"

GravityModule::GravityModule()
 : G_(6.67430e-11)
{}

double GravityModule::computeForce(double m1, double m2, double dist) const {
    if (dist < 1e-6) dist = 1e-6;
    return G_ * (m1*m2)/(dist*dist);
}

cosmos/star.h / star.cpp, black_hole.h / black_hole.cpp, dark_matter.h, etc., each easily 200–2000 lines if you model lifecycles, merging, or advanced physics.

planet_physics/earth.h / earth.cpp would model Earth’s parameters, sub-systems (atmosphere, spin, collisions).

particle_physics.h / .cpp might handle nuclear cross-sections, cosmic ray showers, etc.—again, thousands of lines for real HPC interaction models.

6) Neural Nets: neuralnet/cosmic_nn.h/.cpp + meta_nn.h/.cpp

cosmic_nn.h (expanded for multi-layer, GPU stubs)

#pragma once
#include <vector>

class CosmicNeuralNet {
private:
    struct Layer {
        int inDim, outDim;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::vector<double> outputs;
    };

    std::vector<Layer> layers_;
    bool useGPU_;

public:
    CosmicNeuralNet();

    void initialize(const std::vector<int> &layerDims);
    void enableGPU(bool flag);

    std::vector<double> forward(const std::vector<double> &input);
};

You could expand backprop, RL logic, GPU kernels for forward/backward pass, or use HPC libraries.

meta_nn.h (overseeing the cosmic NN)

#pragma once
#include <vector>
#include "cosmic_nn.h"

class MetaNeuralNetwork {
private:
    std::vector<double> metaParams;
    bool useGPU_;

public:
    MetaNeuralNetwork();

    void initialize(int paramCount);
    void enableGPU(bool flag);
    void adaptCosmicNN(CosmicNeuralNet &net, const std::vector<double> &recentOutputs);
};

In real HPC, you might do distributed training, HPC-based hyperparameter search, etc.

7) Universe Core: universe_core/universe.h/.cpp

universe.h:

#pragma once
#include "../config/cosmic_config.h"
#include "../HPC_core/concurrency/domain_decomposition.h"
#include "../physics/PDE/mhd_solver.h"
#include "../physics/gravity/gravity_module.h"
#include "../physics/planet_physics/earth.h"
#include "../neuralnet/cosmic_nn.h"
#include "../neuralnet/meta_nn.h"
#include "../HPC_core/io_manager/checkpoint_manager.h"
#include "../HPC_core/io_manager/visualization_output.h"
#include "../HPC_core/logging/hpc_logger.h"
#include "../cosmos/star.h"
#include "../cosmos/black_hole.h"
#include "../cosmos/dark_matter.h"
#include <vector>

class Universe {
private:
    HPCConfig &config_;
    HPCLogger logger_;
    DomainDecomposition domDecomp_;
    MHDSolver mhdSolver_;
    GravityModule gravMod_;
    CheckpointManager checkpointMgr_;
    VisualizationOutput vizOutput_;

    // Neural nets
    CosmicNeuralNet cosmicNN_;
    MetaNeuralNetwork metaNN_;

    // HPC concurrency
    int rank_, size_;

    // HPC sim time
    double currentTime_;
    double scaleFactor_;
    double globalEntropy_;

    // HPC cosmic objects
    std::vector<Star> stars_;
    std::vector<BlackHole> blackHoles_;
    std::vector<DarkMatter> darkMatter_;
    Earth planetEarth_;

    // Cache last NN outputs
    std::vector<double> lastNNOutputs_;

public:
    Universe(HPCConfig &cfg);
    void initializeMPI(int &argc, char** &argv);
    void finalizeMPI();
    void initializeUniverse();
    void runSimulation();
    double getCurrentTime() const;

private:
    void evolve(double dt);
    void updateGravity(double dt);
    void updateEarth(double dt);

    void feedNN();
    void interpretNNOutputs();

    void checkpoint(const std::string &filename);

    // Possibly more star formation, BH merges, etc.
    void starFormationCheck();
    void blackHoleMergeCheck();
};

universe.cpp (expanded to thousands of lines if you model PDE coupling, star formation, black hole merging, advanced Earth physics, etc.):

#include "universe.h"
#include <cmath>
#include <mpi.h> // or stubs
#include <iostream>

Universe::Universe(HPCConfig &cfg)
 : config_(cfg),
   rank_(0), size_(1),
   currentTime_(0.0), scaleFactor_(1.0), globalEntropy_(0.0)
{}

void Universe::initializeMPI(int &argc, char** &argv) {
#ifdef HAS_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#endif
    if (rank_==0) {
        logger_.info("Universe HPC environment started: size=" + std::to_string(size_));
    }
}

void Universe::finalizeMPI() {
#ifdef HAS_MPI
    MPI_Finalize();
#endif
}

void Universe::initializeUniverse() {
    // Domain
    int gx = config_.intParams["globalNX"];
    int gy = config_.intParams["globalNY"];
    int gz = config_.intParams["globalNZ"];
    domDecomp_.initialize(rank_, size_, gx, gy, gz);

    // PDE/MHD
    bool useGPU = config_.boolParams["useGPU"];
    mhdSolver_.setLogger(&logger_);
    mhdSolver_.initialize(domDecomp_, rank_, size_, useGPU);

    // NN
    cosmicNN_.initialize({5, 8, 8, 2}); // e.g. 5 input dims
    cosmicNN_.enableGPU(useGPU);
    metaNN_.initialize(5);
    metaNN_.enableGPU(useGPU);

    // Planet Earth
    planetEarth_.setMass(5.972e24); 
    planetEarth_.setPosition(1.496e11, 0, 0); // near 1 AU, simplified

    // Possibly create stars, BH, DM on rank 0
    if (rank_==0) {
        Star s1(1.0e30); 
        s1.setPosition(1.0e10, 0, 0);
        stars_.push_back(s1);

        BlackHole bh1(5.0e30);
        bh1.setPosition(-2.0e10, 0, 0);
        blackHoles_.push_back(bh1);

        // etc...
    }
    logger_.info("Universe initialization complete (rank=" + std::to_string(rank_) + ")");
}

void Universe::runSimulation() {
    double totalTime = config_.numericParams["totalTime"];
    double dt = config_.numericParams["timeStep"];
    int steps = (int)std::ceil(totalTime / dt);

    for (int step=0; step<steps; ++step) {
        evolve(dt);
        feedNN();
        interpretNNOutputs();

        currentTime_ += dt;

        if ((step%50==0) && (rank_==0)) {
            logger_.info("[Universe] step=" + std::to_string(step) +
                         " time=" + std::to_string(currentTime_) +
                         " scale=" + std::to_string(scaleFactor_) +
                         " ent=" + std::to_string(globalEntropy_));
        }

        if ((step%200==0) && (rank_==0)) {
            checkpoint("checkpoint_"+std::to_string(step)+".h5");
        }
    }
    if (rank_==0) {
        logger_.info("[Universe] Simulation finished at time=" + std::to_string(currentTime_));
    }
}

double Universe::getCurrentTime() const {
    return currentTime_;
}

void Universe::checkpoint(const std::string &filename) {
    logger_.info("[Checkpoint] Saving Universe to " + filename);
    // HPC code: gather PDE data, star/earth states, NN weights, etc.
}

void Universe::evolve(double dt) {
    // PDE step
    mhdSolver_.step(dt);

    // Gravity updates
    updateGravity(dt);

    // Earth updates
    updateEarth(dt);

    // comedic cosmic expansion
    scaleFactor_ *= (1.0 + 1e-5*dt);

    // global entropy increase
    globalEntropy_ += 0.01*dt;
}

void Universe::updateGravity(double dt) {
    // gather star, BH, DM, Earth
    // naive O(N^2), or a tree code
    // each Star, BH, DM has position, mass

    // then update positions
    // also check merges, star formation, etc.
}

void Universe::updateEarth(double dt) {
    // Earth rotation, collision events, PDE for atmosphere, placeholders
}

void Universe::feedNN() {
    // PDE data
    double localMag = mhdSolver_.computeLocalMagEnergy();
    double globalMag = localMag;
#ifdef HAS_MPI
    MPI_Allreduce(&localMag, &globalMag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // gather #BH, Earth mass, scaleFactor, etc.
    double bhCount = (double)blackHoles_.size();
    double eMass   = planetEarth_.getMass(); // assumed Earth object has getMass()
    // input= {scaleFactor_, globalEntropy_, globalMag, bhCount, eMass}
    std::vector<double> input = { scaleFactor_, globalEntropy_, globalMag, bhCount, eMass };

    lastNNOutputs_ = cosmicNN_.forward(input);

    // metaNN adapts cosmicNN with recent outputs
    metaNN_.adaptCosmicNN(cosmicNN_, lastNNOutputs_);
}

void Universe::interpretNNOutputs() {
    if (lastNNOutputs_.size()<2) return;
    // outs[0] => tweak scaleFactor
    double sAdj = lastNNOutputs_[0];
    scaleFactor_ *= (1.0 + sAdj*1e-6);

    // outs[1] => if > 0.8 => reduce MHD B fields
    if (lastNNOutputs_[1]>0.8) {
        auto &grid = mhdSolver_.getGrid();
        for (auto &cell : grid) {
            cell.Bx *= 0.9;
            cell.By *= 0.9;
            cell.Bz *= 0.9;
        }
        logger_.info("[Universe] MHD turbulence reduced by NN feedback.");
    }
}

Each of these methods—especially updateGravity, updateEarth—can balloon to thousands of lines once you add:
	•	Tree or FMM for ￼-body.
	•	BH merging / advanced templates.
	•	Star formation PDE coupling.
	•	Earth PDE for atmospheric modeling.
	•	Particle collisions for subnuclear processes.

8) HPC I/O Manager: checkpoint_manager.h/.cpp, visualization_output.h/.cpp

checkpoint_manager.h might show advanced HDF5 usage. You’d have checkpointFields(...), restoreFromCheckpoint(...), etc. easily thousands of lines.

visualization_output.h for in-situ or post-processing frameworks, e.g. VTK or XDMF outputs.

logging/hpc_logger.h/.cpp can handle multi-rank output, concurrency, large logs with thousands of lines for robust HPC usage.

9) Tests in test/ directory

You could have hundreds of test files:
	•	Unit tests for PDE solvers (test_pde_solvers.cpp),
	•	N-Body correctness checks (test_nbody.cpp),
	•	Neural net forward/back checks (test_nn.cpp),
	•	Planet Earth collision logic (test_earth_collisions.cpp),
	•	etc.

Each might be hundreds or thousands of lines** in a serious HPC environment.

Conclusion

The code snippets above show how you can structure a massive HPC cosmic simulation that incorporates:
	•	Planet Earth with placeholders for particle physics collisions or geophysical PDE.
	•	Stars, black holes, dark matter in an HPC-friendly ￼-body approach.
	•	Advanced PDE / MHD modules with domain decomposition, ghost cells, halo exchange.
	•	Neural networks plus a meta-neural net that can adapt or tune the primary network.
	•	MPI + OpenMP concurrency stubs, with GPU placeholders.
	•	Logging, config, checkpointing, in-situ visualization references.

	Scaling to Tens/Hundreds of Thousands of Lines
		•	In actual HPC or research code, each of these files can be hundreds or thousands of lines, especially PDE stencils, HPC concurrency (halo exchange, GPU kernels), and advanced neural net or meta-learning loops.
	•	You can replicate or extend each concept—like partial differential equations for magnetohydrodynamics, advanced gravitational physics, or subnuclear cross-sections for Earth collisions—and balloon the codebase to the “god-tier” scale you desire.

This blueprint is the skeleton. You are now free to copy, paste, expand, and refine each subsystem (PDE stencils, domain decomposition logic, star/BH lifecycle, Earth’s particle collisions, HPC concurrency, etc.) to tens or hundreds of thousands of lines. That is entirely feasible if you fill in every method with domain-specific mathematics, HPC data structures, advanced error handling, parallel load balancing, GPU kernels, in-situ analysis, and more—thus achieving the truly “god-tier” HPC cosmic simulation codebase.what is this code: 


analyze this code:

/***********************************************************************************************
 * universe.cpp
 *
 * A conceptual simulation in C++ that models:
 *   - Multiple Dimensions, each with a discrete TunnelingField
 *   - Simplistic quantum tunneling probabilities influenced by local barrier and decoherence
 *   - A variety of Particle types (Quarks, Bosons, Leptons, Baryons, Aether)
 *   - Basic “Beings” that evolve knowledge over time
 *   - A rudimentary QuantumCommunication mechanism to transmit messages between Dimensions
 *
 * Disclaimer:
 * This code is provided purely as an illustrative, academic demonstration. It does not claim to
 * represent accurate physics or mathematical models of the universe. All parameter values and
 * equations are conceptual in nature.
 *
 * Compilation (with optional OpenMP support):
 *   g++ universe.cpp -fopenmp -o universe
 ***********************************************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#ifdef _OPENMP
  #include <omp.h>
#endif

//------------------------------------------------------------
// 0) Enumerations, Utility Functions, and Forward Declarations
//------------------------------------------------------------

enum class ParticleType {
    QUARK,
    BOSON,
    AETHER,
    LEPTON,   // For electrons, muons, etc.
    BARYON,   // For protons, neutrons
    UNKNOWN
};

enum class QuarkFlavor {
    UP,
    DOWN,
    CHARM,
    STRANGE,
    TOP,
    BOTTOM
};

enum class BosonType {
    PHOTON,
    GLUON,
    W_PLUS,
    W_MINUS,
    Z,
    HIGGS
};

enum class DimState {
    HEAVEN,
    HELL,
    NORMAL,
    PDE_REALM
};

// Clamping utility for numeric values
double clampDouble(double val, double minVal, double maxVal) {
    return std::max(minVal, std::min(maxVal, val));
}

// Forward declarations
class Dimension;
class MatrixEngine;

//--------------------------------------------------------------------------
// EquationRepository: A conceptual structure for PDE or quantum equations
//--------------------------------------------------------------------------
class EquationRepository {
public:
    EquationRepository() {
        m_equations["default"]      = "E = m*c^2";
        m_equations["pde_wave"]     = "phi_next = 2*phi - phi_prev + c^2*(laplacian(phi))";
        m_equations["quantum_flux"] = "Psi' = i*(hbar)*laplacian(Psi) - U*Psi";
    }

    std::string getEquation(const std::string &name) const {
        auto it = m_equations.find(name);
        return (it != m_equations.end()) ? it->second : "UNDEFINED_EQUATION";
    }

    void setEquation(const std::string &name, const std::string &eq) {
        m_equations[name] = eq;
    }

    void simulateEquation(const std::string &name) const {
        // Placeholder for conceptual simulation
        std::cout << "[EquationRepository] Simulating: " << getEquation(name) << "\n";
    }

private:
    std::map<std::string, std::string> m_equations;
};

//--------------------------------------------------------------------
// 1) Particle Base Class + Derived Quark, Boson, Aether, etc.
//--------------------------------------------------------------------
class Particle {
public:
    Particle(ParticleType type, double mass, double charge, double spin,
             double x=0.0, double y=0.0, double z=0.0,
             double px=0.0, double py=0.0, double pz=0.0)
    : m_type(type),
      m_mass(mass),
      m_charge(charge),
      m_spin(spin),
      m_x(x), m_y(y), m_z(z),
      m_px(px), m_py(py), m_pz(pz)
    { }

    virtual ~Particle() = default;

    // Basic classical update
    virtual void update(double dt) {
        double vx = (m_mass > 0.0) ? (m_px / m_mass) : 0.0;
        double vy = (m_mass > 0.0) ? (m_py / m_mass) : 0.0;
        double vz = (m_mass > 0.0) ? (m_pz / m_mass) : 0.0;

        m_x += vx * dt;
        m_y += vy * dt;
        m_z += vz * dt;
    }

    // Accessors
    ParticleType type() const { return m_type; }
    double mass()      const { return m_mass; }
    double charge()    const { return m_charge; }
    double spin()      const { return m_spin; }

    double x()   const { return m_x; }
    double y()   const { return m_y; }
    double z()   const { return m_z; }
    double px()  const { return m_px; }
    double py()  const { return m_py; }
    double pz()  const { return m_pz; }

    void addMomentum(double dpx, double dpy, double dpz) {
        m_px += dpx;
        m_py += dpy;
        m_pz += dpz;
    }

    virtual std::string info() const {
        return "ParticleBase";
    }

protected:
    ParticleType m_type;
    double       m_mass;   // In MeV/c^2 (conceptual example)
    double       m_charge; // In elementary charge units
    double       m_spin;   // Spin in ℏ units

    double m_x, m_y, m_z;      // Position
    double m_px, m_py, m_pz;   // Momentum
};

//--------------------------------------------------------------------
// Quark
class Quark : public Particle {
public:
    Quark(QuarkFlavor flavor,
          double x=0.0, double y=0.0, double z=0.0,
          double px=0.0, double py=0.0, double pz=0.0)
    : Particle(ParticleType::QUARK,
               getMass(flavor),
               getCharge(flavor),
               0.5,  // spin 1/2
               x, y, z, px, py, pz),
      m_flavor(flavor)
    { }

    std::string info() const override {
        return "Quark(" + flavorToString(m_flavor) + ")";
    }

private:
    QuarkFlavor m_flavor;

    static double getMass(QuarkFlavor f) {
        // Approximate quark masses in MeV
        switch(f) {
            case QuarkFlavor::UP:      return 2.3;
            case QuarkFlavor::DOWN:    return 4.8;
            case QuarkFlavor::CHARM:   return 1275.0;
            case QuarkFlavor::STRANGE: return 95.0;
            case QuarkFlavor::TOP:     return 173100.0;
            case QuarkFlavor::BOTTOM:  return 4180.0;
        }
        return 0.0;
    }

    static double getCharge(QuarkFlavor f) {
        switch(f) {
            case QuarkFlavor::UP:
            case QuarkFlavor::CHARM:
            case QuarkFlavor::TOP:
                return +2.0/3.0;
            case QuarkFlavor::DOWN:
            case QuarkFlavor::STRANGE:
            case QuarkFlavor::BOTTOM:
                return -1.0/3.0;
        }
        return 0.0;
    }

    static std::string flavorToString(QuarkFlavor f) {
        switch(f) {
            case QuarkFlavor::UP:      return "Up";
            case QuarkFlavor::DOWN:    return "Down";
            case QuarkFlavor::CHARM:   return "Charm";
            case QuarkFlavor::STRANGE: return "Strange";
            case QuarkFlavor::TOP:     return "Top";
            case QuarkFlavor::BOTTOM:  return "Bottom";
        }
        return "UnknownQuarkFlavor";
    }
};

//--------------------------------------------------------------------
// Boson
class Boson : public Particle {
public:
    Boson(BosonType bosonType,
          double x=0.0, double y=0.0, double z=0.0,
          double px=0.0, double py=0.0, double pz=0.0)
    : Particle(ParticleType::BOSON,
               getMass(bosonType),
               getCharge(bosonType),
               getSpin(bosonType),
               x, y, z, px, py, pz),
      m_bosonType(bosonType)
    { }

    std::string info() const override {
        return "Boson(" + bosonToString(m_bosonType) + ")";
    }

private:
    BosonType m_bosonType;

    static double getMass(BosonType b) {
        // Approximate boson masses in MeV
        switch(b) {
            case BosonType::PHOTON:   return 0.0;
            case BosonType::GLUON:    return 0.0;
            case BosonType::W_PLUS:
            case BosonType::W_MINUS:  return 80379.0;
            case BosonType::Z:        return 91187.0;
            case BosonType::HIGGS:    return 125100.0;
        }
        return 0.0;
    }

    static double getCharge(BosonType b) {
        switch(b) {
            case BosonType::PHOTON:  return 0.0;
            case BosonType::GLUON:   return 0.0;
            case BosonType::W_PLUS:  return +1.0;
            case BosonType::W_MINUS: return -1.0;
            case BosonType::Z:       return 0.0;
            case BosonType::HIGGS:   return 0.0;
        }
        return 0.0;
    }

    static double getSpin(BosonType b) {
        switch(b) {
            case BosonType::PHOTON:
            case BosonType::GLUON:
            case BosonType::W_PLUS:
            case BosonType::W_MINUS:
            case BosonType::Z:
                return 1.0;
            case BosonType::HIGGS:
                return 0.0;
        }
        return 0.0;
    }

    static std::string bosonToString(BosonType b) {
        switch(b) {
            case BosonType::PHOTON:   return "Photon";
            case BosonType::GLUON:    return "Gluon";
            case BosonType::W_PLUS:   return "W+";
            case BosonType::W_MINUS:  return "W-";
            case BosonType::Z:        return "Z";
            case BosonType::HIGGS:    return "Higgs";
        }
        return "UnknownBoson";
    }
};

//--------------------------------------------------------------------
// Aether
class Aether : public Particle {
public:
    Aether(double x=0.0, double y=0.0, double z=0.0,
           double px=0.0, double py=0.0, double pz=0.0)
    : Particle(ParticleType::AETHER,
               1e-3,
               0.0,
               0.0,
               x, y, z, px, py, pz)
    { }

    std::string info() const override {
        return "Aether(DivineParticle)";
    }
};

//--------------------------------------------------------------------
// Electron: a Lepton
class Electron : public Particle {
public:
    Electron(double x=0.0, double y=0.0, double z=0.0,
             double px=0.0, double py=0.0, double pz=0.0)
    : Particle(ParticleType::LEPTON,
               0.511,     // Approximate mass in MeV
               -1.0,      // Electric charge
               0.5,       // Spin
               x, y, z, px, py, pz)
    { }

    std::string info() const override {
        return "Electron";
    }
};

//--------------------------------------------------------------------
// Proton: a Baryon
class Proton : public Particle {
public:
    Proton(double x=0.0, double y=0.0, double z=0.0,
           double px=0.0, double py=0.0, double pz=0.0)
    : Particle(ParticleType::BARYON,
               938.272,   // Approximate mass in MeV
               +1.0,      // Electric charge
               0.5,       // Spin
               x, y, z, px, py, pz)
    { }

    std::string info() const override {
        return "Proton";
    }
};

//--------------------------------------------------------------------
// Neutron: a Baryon
class Neutron : public Particle {
public:
    Neutron(double x=0.0, double y=0.0, double z=0.0,
            double px=0.0, double py=0.0, double pz=0.0)
    : Particle(ParticleType::BARYON,
               939.565,   // Approximate mass in MeV
               0.0,       // Neutral
               0.5,
               x, y, z, px, py, pz)
    { }

    std::string info() const override {
        return "Neutron";
    }
};

//--------------------------------------------------------------------
// 2) Being class -- naive “consciousness” or agent
//--------------------------------------------------------------------
class Being {
public:
    Being(std::string name, double knowledgeLevel = 0.5)
        : m_name(name), m_knowledgeLevel(knowledgeLevel)
    { }

    std::string name() const { return m_name; }
    double knowledgeLevel() const { return m_knowledgeLevel; }

    void interpretObservations() {
        if (m_knowledgeLevel < 0.3) {
            std::cout << "[Being " << m_name << "] is significantly hindered by interpretative biases.\n";
        } else if (m_knowledgeLevel < 0.7) {
            std::cout << "[Being " << m_name << "] partially discerns underlying phenomena.\n";
        } else {
            std::cout << "[Being " << m_name << "] demonstrates a highly coherent understanding.\n";
        }
    }

    void growKnowledge(double amount) {
        m_knowledgeLevel = clampDouble(m_knowledgeLevel + amount, 0.0, 1.0);
    }

private:
    std::string m_name;
    double      m_knowledgeLevel;
};

//--------------------------------------------------------------------
// Tunneling mechanics (barrier, probability, decoherence, etc.)
//--------------------------------------------------------------------
struct TunnelingParams {
    double barrierHeight;
    double barrierWidth;
};

class QuantumTunneling {
public:
    static bool attemptTunneling(const Particle &p, const TunnelingParams &tp, double decoherenceFactor) {
        double energy = getParticleEnergy(p);

        // If the particle energy exceeds the barrier height, tunneling occurs trivially
        if (energy > tp.barrierHeight) {
            return true;
        }

        // Probability decays with barrier width, also incorporating a decoherence factor
        double alpha = 1.0 + decoherenceFactor;
        double prob = std::exp(-alpha * tp.barrierWidth);

        // Scale probability by the ratio of particle energy to barrier height
        double ratio = energy / (tp.barrierHeight + 1e-9);
        double finalProb = prob * ratio;

        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double roll = dist(rng);

        return (roll < finalProb);
    }

private:
    static double getParticleEnergy(const Particle &p) {
        // A simplistic kinetic energy approximation: KE = 0.5 * m * v^2
        double m = p.mass() + 1e-9;
        double vx = p.px() / m;
        double vy = p.py() / m;
        double vz = p.pz() / m;
        double speedSq = vx*vx + vy*vy + vz*vz;
        return 0.5 * p.mass() * speedSq;
    }
};

//--------------------------------------------------------------------
// 3) TunnelingField: a discrete grid in each dimension
//--------------------------------------------------------------------
class TunnelingField {
public:
    TunnelingField(int w, int h)
    : m_width(w), m_height(h)
    {
        // Initialize random barrier and decoherence values
        m_barrierMap.resize(w*h);
        m_decoherenceMap.resize(w*h);

        std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> distBarrier(10.0, 80.0);
        std::uniform_real_distribution<double> distDeco(0.0, 0.5);

        for (int i = 0; i < w*h; ++i) {
            m_barrierMap[i] = distBarrier(rng);
            m_decoherenceMap[i] = distDeco(rng);
        }
    }

    double localBarrierHeight(int x, int y) const {
        if (x < 0 || y < 0 || x >= m_width || y >= m_height) return 80.0; 
        return m_barrierMap[y*m_width + x];
    }
    double localDecoherence(int x, int y) const {
        if (x < 0 || y < 0 || x >= m_width || y >= m_height) return 0.5;
        return m_decoherenceMap[y*m_width + x];
    }

    int width() const  { return m_width; }
    int height() const { return m_height; }

    void evolveField(double dt) {
        // Example evolution by random fluctuation
        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> noise(-1.0,1.0);
        for (int i = 0; i < m_width*m_height; ++i) {
            m_barrierMap[i] += noise(rng)*0.1 * dt;
            m_barrierMap[i] = clampDouble(m_barrierMap[i],10.0,100.0);

            m_decoherenceMap[i] += noise(rng)*0.01 * dt;
            m_decoherenceMap[i] = clampDouble(m_decoherenceMap[i],0.0,1.0);
        }
    }

private:
    int m_width, m_height;
    std::vector<double> m_barrierMap;
    std::vector<double> m_decoherenceMap;
};

//--------------------------------------------------------------------
// 4) Dimension class with a TunnelingField
//--------------------------------------------------------------------
class Dimension {
public:
    Dimension(std::string name, DimState state, double baseEnergyScale, int fieldW=10, int fieldH=10)
        : m_name(name),
          m_state(state),
          m_baseEnergyScale(baseEnergyScale),
          m_field(fieldW, fieldH)
    {
        m_globalParams.barrierHeight = 50.0;
        m_globalParams.barrierWidth  = 1.0;
    }

    void setGlobalTunnelingParams(double barrierH, double barrierW) {
        m_globalParams.barrierHeight = barrierH;
        m_globalParams.barrierWidth  = barrierW;
    }

    void addParticle(std::unique_ptr<Particle> p) {
        m_particles.push_back(std::move(p));
    }
    void addBeing(Being b) {
        m_beings.push_back(b);
    }

    void step(double dt) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i=0; i<(int)m_particles.size(); ++i) {
            m_particles[i]->update(dt);
        }

        // Apply dimension-specific influences
        switch(m_state) {
            case DimState::HELL:
                applyHellEffects(dt);
                break;
            case DimState::HEAVEN:
                applyHeavenEffects(dt);
                break;
            case DimState::NORMAL:
                // Minimal additional effects
                break;
            case DimState::PDE_REALM:
                applyPDERules(dt);
                break;
        }

        // Evaluate knowledge changes in Beings
        evaluateKnowledgeState(dt);

        // Evolve the local TunnelingField
        m_field.evolveField(dt);

        // Apply localized tunneling
        applyLocalQuantumTunneling(dt);
    }

    void printState() const {
        std::cout << "[Dimension: " << m_name << "] - " << stateToString(m_state)
                  << ", baseEnergy=" << m_baseEnergyScale << "\n";
        std::cout << "GlobalBarrier(H="<<m_globalParams.barrierHeight
                  <<",W="<<m_globalParams.barrierWidth<<")\n";
        std::cout << "Particles:\n";
        for (size_t i=0; i<m_particles.size(); ++i) {
            auto &p = m_particles[i];
            std::cout << "  #" << i << ": " << p->info()
                      << " pos=(" << p->x() <<","<< p->y() <<","<< p->z()<<")"
                      << " p=(" << p->px()<<","<< p->py()<<","<<p->pz()<<")\n";
        }
        std::cout << "Beings:\n";
        for (auto &b : m_beings) {
            std::cout << "  [Being] " << b.name()
                      << ", knowledge=" << b.knowledgeLevel()<<"\n";
        }
        std::cout << "--------------------------------------\n";
    }

    // Cross-dimension usage
    std::unique_ptr<Particle> removeParticleAtIndex(size_t idx) {
        if (idx >= m_particles.size()) return nullptr;
        std::unique_ptr<Particle> ret = std::move(m_particles[idx]);
        m_particles.erase(m_particles.begin()+idx);
        return ret;
    }
    size_t particleCount() const { return m_particles.size(); }
    size_t beingCount()   const { return m_beings.size(); }
    std::string name()    const { return m_name; }

private:
    std::string             m_name;
    DimState                m_state;
    double                  m_baseEnergyScale;
    std::vector<std::unique_ptr<Particle>> m_particles;
    std::vector<Being>      m_beings;

    TunnelingField          m_field;
    TunnelingParams         m_globalParams;

    void evaluateKnowledgeState(double dt) {
        for (auto &b : m_beings) {
            b.interpretObservations();
            b.growKnowledge(0.01*dt);
        }
    }

    void applyHellEffects(double dt) {
        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-2.0,2.0);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i=0; i<(int)m_particles.size(); ++i){
            double dpx = dist(rng)*dt*m_baseEnergyScale;
            double dpy = dist(rng)*dt*m_baseEnergyScale;
            double dpz = dist(rng)*dt*m_baseEnergyScale;
            m_particles[i]->addMomentum(dpx,dpy,dpz);
        }
    }

    void applyHeavenEffects(double dt) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i=0; i<(int)m_particles.size(); ++i){
            auto &p = m_particles[i];
            double factor=0.95;
            double newPx=p->px()*factor;
            double newPy=p->py()*factor;
            double newPz=p->pz()*factor;
            newPx += 0.01*dt; // A small, conceptual offset
            p->addMomentum(newPx-p->px(), newPy-p->py(), newPz-p->pz());
        }
    }

    void applyPDERules(double dt) {
        // Conceptual PDE influence
        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-0.5,0.5);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i=0; i<(int)m_particles.size(); ++i){
            double wave = dist(rng)*dt*m_baseEnergyScale;
            m_particles[i]->addMomentum(0, 0, wave);
        }
    }

    void applyLocalQuantumTunneling(double /*dt*/) {
        // Evaluate local field and attempt tunneling
        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<int> cDistX(0, m_field.width()-1);
        std::uniform_int_distribution<int> cDistY(0, m_field.height()-1);

        for (auto &p : m_particles) {
            int cx = (int)std::floor( (p->x()+50.0)/10.0 );
            int cy = (int)std::floor( (p->y()+50.0)/10.0 );

            double localBarrier = m_field.localBarrierHeight(cx,cy);
            double localDeco    = m_field.localDecoherence(cx,cy);

            TunnelingParams combined;
            combined.barrierHeight = (localBarrier + m_globalParams.barrierHeight)*0.5;
            combined.barrierWidth  = m_globalParams.barrierWidth;

            bool success = QuantumTunneling::attemptTunneling(*p, combined, localDeco);
            if(success) {
                int nx = cDistX(rng);
                int ny = cDistY(rng);
                double newX = nx*10.0 - 50.0;
                double newY = ny*10.0 - 50.0;
                double newZ = p->z();

                shiftParticle(*p, newX, newY, newZ);
                std::cout << "[LocalTunnel] " << p->info() << " => new cell("
                          << nx <<","<<ny<<") pos("<<newX<<","<<newY<<","<<newZ<<")\n";
            }
        }
    }

    // Helper to shift a particle’s position without re-instantiating
    void shiftParticle(Particle &p, double nx, double ny, double nz) {
        struct Hack : Particle {
            using Particle::m_x; using Particle::m_y; using Particle::m_z;
            Hack(Particle* p) : Particle(*p) {}
        };
        Hack* h = reinterpret_cast<Hack*>(&p);
        h->m_x = nx; h->m_y = ny; h->m_z = nz;
    }

    static std::string stateToString(DimState s) {
        switch(s) {
            case DimState::HEAVEN:    return "HEAVEN";
            case DimState::HELL:      return "HELL";
            case DimState::NORMAL:    return "NORMAL";
            case DimState::PDE_REALM: return "PDE_REALM";
        }
        return "UNKNOWN_DIMSTATE";
    }
};

//--------------------------------------------------------------------
// 5) A naive "QuantumCommunication" system for inter-dimensional messaging
//--------------------------------------------------------------------
struct QuantumMessage {
    std::string data;
    bool        encrypted;
};

class QuantumCommunication {
public:
    static bool sendMessage(Dimension &src, Dimension &dst, const QuantumMessage &msg) {
        // Simplistic probability-based message sending
        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0,1.0);

        double prob = 0.5; // Base probability
        if (src.name()!=dst.name()) {
            prob += 0.2; 
        }
        double roll = dist(rng);
        if (roll < prob) {
            std::cout<<"[QComm] Message delivered from "<<src.name()
                     <<" to "<<dst.name()<<": \""<<msg.data<<"\"\n";
            return true;
        }
        else {
            std::cout<<"[QComm] Message FAILED to tunnel from "<<src.name()
                     <<" to "<<dst.name()<<"\n";
            return false;
        }
    }
};

//--------------------------------------------------------------------
// 6) MatrixEngine orchestrates multiple Dimensions
//--------------------------------------------------------------------
class MatrixEngine {
public:
    MatrixEngine() {
        eqRepo.setEquation("custom_law","E_new = E_old + PDE_term(...)");
    }

    Dimension& createDimension(const std::string &name, DimState state,
                               double baseEnergyScale,
                               int fieldW=10, int fieldH=10)
    {
        m_dimensions.emplace_back(std::make_unique<Dimension>(
            name, state, baseEnergyScale, fieldW, fieldH
        ));
        return *m_dimensions.back();
    }

    void stepAll(double dt) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i=0; i<(int)m_dimensions.size();++i){
            m_dimensions[i]->step(dt);
        }
        crossDimSynchronicities(dt);
        eqRepo.simulateEquation("default");
    }

    void printAllDimensions() const {
        std::cout << "\n=== MatrixEngine: Multi-Dimension Snapshot ===\n";
        for(auto &dimPtr: m_dimensions){
            dimPtr->printState();
        }
        std::cout << "=============================================\n";
    }

    Dimension& getDimension(int i) { return *m_dimensions[i]; }
    int dimensionCount() const { return (int)m_dimensions.size(); }

private:
    std::vector<std::unique_ptr<Dimension>> m_dimensions;
    EquationRepository eqRepo;

    void crossDimSynchronicities(double dt) {
        // Occasionally shift a random particle between two random Dimensions
        if(m_dimensions.size()<2)return;
        static std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<double> probDist(0.0,1.0);

        if(probDist(rng)<0.02) {
            std::uniform_int_distribution<int> idxDist(0,(int)m_dimensions.size()-1);
            int s=idxDist(rng), d=idxDist(rng);
            if(d==s && m_dimensions.size()>1) d=(d+1)%m_dimensions.size();

            Dimension &src= *m_dimensions[s];
            Dimension &dst= *m_dimensions[d];

            if(src.particleCount()>0){
                std::uniform_int_distribution<int> pDist(0,(int)src.particleCount()-1);
                int pIdx=pDist(rng);
                auto p=src.removeParticleAtIndex(pIdx);
                if(p){
                    p->addMomentum(0,5.0*dt,0);
                    dst.addParticle(std::move(p));
                    std::cout<<"[SYNC] Cross-dimensional shift from "<<src.name()
                             <<" to "<<dst.name()<<"\n";
                }
            }
        }
    }
};

//--------------------------------------------------------------------
// 7) Demo main: Illustrating the system with additional particle types
//--------------------------------------------------------------------
int main(){
    std::cout<<"=== universe.cpp: Discrete TunnelingField & Basic QuantumCommunication ===\n";

    MatrixEngine engine;

    // Create Dimensions
    Dimension &dimHeaven = engine.createDimension("Heaven", DimState::HEAVEN, 0.5, 8, 8);
    Dimension &dimHell   = engine.createDimension("Hell",   DimState::HELL,   2.0,10,10);
    Dimension &dimEarth  = engine.createDimension("Earth",  DimState::NORMAL, 1.0,12,12);

    // Set global tunneling parameters
    dimHeaven.setGlobalTunnelingParams(20.0,1.0);
    dimHell.setGlobalTunnelingParams(90.0,2.5);
    dimEarth.setGlobalTunnelingParams(50.0,1.0);

    // Populate the dimensions with various Particles and Beings
    // Heaven
    dimHeaven.addParticle(std::make_unique<Boson>(BosonType::PHOTON));
    dimHeaven.addParticle(std::make_unique<Aether>(0.0,0.0,0.0));
    dimHeaven.addBeing(Being("Angel",0.8));

    // Hell
    dimHell.addParticle(std::make_unique<Quark>(QuarkFlavor::DOWN,-1.0,0.0,0.0));
    dimHell.addBeing(Being("Demon",0.2));

    // Earth
    dimEarth.addParticle(std::make_unique<Quark>(QuarkFlavor::UP,2.0,2.0,0.0));
    dimEarth.addParticle(std::make_unique<Boson>(BosonType::W_MINUS,1.0,1.0,0.0,2.0,0.0,0.0));
    dimEarth.addBeing(Being("Human",0.5));

    // Additional: Electrons, Protons, Neutrons
    dimEarth.addParticle(std::make_unique<Electron>(5.0, 5.0, 0.0));
    dimEarth.addParticle(std::make_unique<Proton>(-2.0, 3.0, 0.0));
    dimEarth.addParticle(std::make_unique<Neutron>(-1.0, -4.0, 0.0));

    // Run simulation steps
    for(int step=0; step<5; ++step){
        std::cout<<"\nTime Step "<<step<<":\n";
        engine.printAllDimensions();
        engine.stepAll(0.1);

        // Attempt quantum communication at step = 2
        if(step==2){
            QuantumMessage msg{"Hello from Earth!", true};
            auto &earth  = engine.getDimension(2); // Earth
            auto &heaven = engine.getDimension(0); // Heaven
            QuantumCommunication::sendMessage(earth, heaven, msg);
        }
    }

    std::cout<<"=== Simulation end. ===\n";
    return 0;
}