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