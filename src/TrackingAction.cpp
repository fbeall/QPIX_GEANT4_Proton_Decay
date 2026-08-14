// -----------------------------------------------------------------------------
//  TrackingAction.cpp
//
//
//   * Author: Everybody is an author!
//   * Creation date: 4 August 2020
// -----------------------------------------------------------------------------

#include "TrackingAction.h"

// Q-Pix includes
#include "MCParticle.h"
#include "MCTruthManager.h"

// GEANT4 includes
#include "G4TrackingManager.hh"

// C++ includes
#include <iostream>
#include <fstream>

#include "G4VProcess.hh"

#include "ConfigManager.h"

#include <cmath>

TrackingAction::TrackingAction()
{}

TrackingAction::~TrackingAction()
{}

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
    // get MC truth manager
    MCTruthManager * mc_truth_manager = MCTruthManager::Instance();

    // create new MCParticle object
    MCParticle * particle = new MCParticle();
    particle->SetTrackID(track->GetTrackID());
    particle->SetParentTrackID(track->GetParentID());
    particle->SetPDGCode(track->GetDefinition()->GetPDGEncoding());
    particle->SetMass(track->GetDynamicParticle()->GetMass());
    particle->SetCharge(track->GetDynamicParticle()->GetCharge());
    particle->SetGlobalTime(track->GetGlobalTime() / CLHEP::ns);
    particle->SetTotalOccupancy(track->GetDynamicParticle()->GetTotalOccupancy());

    particle->SetInitialPosition(
        TLorentzVector(
            track->GetPosition().x() / CLHEP::cm,
            track->GetPosition().y() / CLHEP::cm,
            track->GetPosition().z() / CLHEP::cm,
            track->GetGlobalTime()   / CLHEP::ns
        )
    );

    particle->SetInitialMomentum(
        TLorentzVector(
            track->GetMomentum().x() / CLHEP::MeV,
            track->GetMomentum().y() / CLHEP::MeV,
            track->GetMomentum().z() / CLHEP::MeV,
            track->GetTotalEnergy()  / CLHEP::MeV
        )
    );

    // add track ID to parent MC particle
    // we might need to deal with cases where some particles aren't tracked (?)
    // we can use a try block for that if need be
    if (track->GetParentID() > 0)
    {
        // get parent MC particle
        MCParticle * parent_particle = mc_truth_manager->GetMCParticle(track->GetParentID());
        parent_particle->AddDaughter(track->GetTrackID());
    }

    // add MC particle to MC truth manager
    mc_truth_manager->AddMCParticle(particle);
    //G4cout << "Added MCParticle" << G4endl;
}

void TrackingAction::PostUserTrackingAction(const G4Track* track)
{
    // get MC truth manager
    MCTruthManager * mc_truth_manager = MCTruthManager::Instance();

    // get MC particle
    MCParticle * particle = mc_truth_manager->GetMCParticle(track->GetTrackID());

    // Storing end-of-life particle info when the track ends (FB 8-14-26)
    const G4Step* lastStep = track->GetStep();

    if (lastStep == nullptr || lastStep->GetPostStepPoint() == nullptr) // prevents crash if G4 pops out a track with no final step
    {
        particle->SetProcess("NoProcess");
        particle->SetDecayed(false);
        particle->SetDetectorXTag(-1);
        particle->SetDetectorYTag(-1);
        return;
    }

    const G4StepPoint* post = lastStep->GetPostStepPoint();
    const G4VProcess* process = post->GetProcessDefinedStep();

    G4String processName = process ? process->GetProcessName() : "User Limit";

    bool decayed =
        processName == "Decay" ||
        processName == "RadioactiveDecay" ||
        processName == "RadioactiveDecayBase";

    particle->SetFinalPosition(
        TLorentzVector(
            post->GetPosition().x() / CLHEP::cm,
            post->GetPosition().y() / CLHEP::cm,
            post->GetPosition().z() / CLHEP::cm,
            post->GetGlobalTime()   / CLHEP::ns
        )
    );

    // Storing pixel/detector tags
    //particle->SetFinalTime(...);
    particle->SetProcess(processName);
    particle->SetDecayed(decayed);

    double const final_x_cm = post->GetPosition().x() / CLHEP::cm;
    double const final_y_cm = post->GetPosition().y() / CLHEP::cm;

    double const pixel_size_cm = 0.4;  // Must match Q_PIX_RTD pixel_size metadata, in cm.

    int detector_x_tag = -1;
    int detector_y_tag = -1;

    if (final_x_cm >= 0. &&
        final_x_cm < ConfigManager::GetDetectorWidth() / CLHEP::cm &&
        final_y_cm >= 0. &&
        final_y_cm < ConfigManager::GetDetectorHeight() / CLHEP::cm)
    {
        detector_x_tag = static_cast<int>(std::floor(final_x_cm / pixel_size_cm));
        detector_y_tag = static_cast<int>(std::floor(final_y_cm / pixel_size_cm));
    }

    particle->SetDetectorXTag(detector_x_tag);
    particle->SetDetectorYTag(detector_y_tag);

    // set process
    // if (track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep() != 0) {
    //   particle->SetProcess(track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName());
    // } else {
    //   particle->SetProcess("User Limit");
    // }
}

