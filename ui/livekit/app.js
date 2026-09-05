/*
 * ============================================================
 * VoiceGuard - LiveKit Client
 * ============================================================
 *
 * Expected URL:
 *
 * index.html?url=<LIVEKIT_URL>&token=<JWT>
 *
 * Example:
 *
 * index.html?url=wss://your-project.livekit.cloud&token=eyJ...
 *
 * The LiveKit API secret must NEVER be placed in this file.
 * ============================================================
 */


/* ============================================================
   URL PARAMETERS
   ============================================================ */

const params = new URLSearchParams(window.location.search);

const LIVEKIT_URL =
    params.get("url") ||
    window.VOICEGUARD_LIVEKIT_URL;

const TOKEN =
    params.get("token") ||
    window.VOICEGUARD_LIVEKIT_TOKEN;

/* ============================================================
   DOM ELEMENTS
   ============================================================ */

const participantsContainer =
    document.getElementById("participants");

const statusText =
    document.getElementById("status-text");

const statusDot =
    document.getElementById("status-dot");

const muteButton =
    document.getElementById("mute-btn");

const leaveButton =
    document.getElementById("leave-btn");


/* ============================================================
   STATE
   ============================================================ */

let room = null;

let microphoneEnabled = true;


/* ============================================================
   STATUS
   ============================================================ */

function setStatus(text, live = false) {

    if (statusText) {
        statusText.textContent = text;
    }

    if (statusDot) {
        statusDot.classList.toggle("live", live);
    }
}


/* ============================================================
   EMPTY STATE
   ============================================================ */

function removeEmptyState() {

    const emptyState =
        document.getElementById("empty-state");

    if (emptyState) {
        emptyState.remove();
    }
}


function showEmptyState() {

    /*
     * Don't create duplicates.
     */

    if (
        document.getElementById("empty-state")
    ) {
        return;
    }

    const empty =
        document.createElement("div");

    empty.id = "empty-state";
    empty.className = "participant";

    empty.innerHTML = `
        <div>
            <div class="avatar">
                👤
            </div>

            <div class="message">
                Waiting for participant...
            </div>
        </div>
    `;

    participantsContainer.appendChild(empty);
}


/* ============================================================
   PARTICIPANT CARD
   ============================================================ */

function createParticipantCard(participant) {

    const participantId =
        "participant-" +
        participant.identity;

    /*
     * Don't create duplicate cards.
     */

    const existing =
        document.getElementById(
            participantId
        );

    if (existing) {
        return existing;
    }


    removeEmptyState();


    const card =
        document.createElement("div");

    card.className = "participant";

    card.id = participantId;


    /*
     * Avatar
     */

    const avatar =
        document.createElement("div");

    avatar.className = "avatar";

    avatar.textContent = "👤";


    /*
     * Participant name
     */

    const name =
        document.createElement("div");

    name.className =
        "participant-name";

    name.textContent =
        participant.name ||
        participant.identity;


    card.appendChild(avatar);

    card.appendChild(name);

    participantsContainer.appendChild(card);


    return card;
}


/* ============================================================
   REMOVE PARTICIPANT
   ============================================================ */

function removeParticipantCard(participant) {

    const participantId =
        "participant-" +
        participant.identity;

    const card =
        document.getElementById(
            participantId
        );

    if (card) {
        card.remove();
    }


    /*
     * If no participants remain,
     * show the waiting state.
     */

    const cards =
        participantsContainer
            .querySelectorAll(
                ".participant"
            );

    if (cards.length === 0) {
        showEmptyState();
    }
}


/* ============================================================
   ATTACH TRACK
   ============================================================ */

function attachTrack(track, participant) {

    const card =
        createParticipantCard(
            participant
        );


    const element =
        track.attach();


    /*
     * VIDEO
     */

    if (
        track.kind ===
        LivekitClient.Track.Kind.Video
    ) {

        element.className =
            "participant-video";

        element.autoplay = true;

        element.playsInline = true;

        /*
         * Put video behind the
         * participant name.
         */

        card.insertBefore(
            element,
            card.firstChild
        );
    }


    /*
     * AUDIO
     */

    else if (
        track.kind ===
        LivekitClient.Track.Kind.Audio
    ) {

        /*
         * Audio doesn't need
         * a visual element.
         *
         * Keep it attached to
         * the document so the
         * browser plays it.
         */

        element.autoplay = true;

        element.style.display =
            "none";

        document.body.appendChild(
            element
        );
    }
}


/* ============================================================
   DETACH TRACK
   ============================================================ */

function detachTrack(track) {

    try {

        track.detach().forEach(
            element => {
                element.remove();
            }
        );

    } catch (error) {

        console.warn(
            "Could not detach track:",
            error
        );
    }
}


/* ============================================================
   ADD EXISTING PARTICIPANT TRACKS
   ============================================================ */

function setupExistingParticipant(
    participant
) {

    createParticipantCard(
        participant
    );


    /*
     * Publications already
     * subscribed to.
     */

    participant.trackPublications
        .forEach(
            publication => {

                if (
                    publication.track
                ) {

                    attachTrack(
                        publication.track,
                        participant
                    );
                }
            }
        );
}


/* ============================================================
   SPEAKING INDICATOR
   ============================================================ */

function updateSpeakingIndicators(
    speakers
) {

    /*
     * Remove speaking state
     * from everyone first.
     */

    document
        .querySelectorAll(
            ".participant"
        )
        .forEach(
            card => {

                card.classList.remove(
                    "speaking"
                );
            }
        );


    /*
     * Highlight currently
     * active speakers.
     */

    speakers.forEach(
        speaker => {

            const participantId =
                "participant-" +
                speaker.identity;

            const card =
                document.getElementById(
                    participantId
                );

            if (card) {

                card.classList.add(
                    "speaking"
                );
            }
        }
    );
}


/* ============================================================
   CONNECT TO LIVEKIT
   ============================================================ */

async function connectToRoom() {

    /*
     * Check that LiveKit SDK
     * has loaded.
     */

    if (
        typeof LivekitClient ===
        "undefined"
    ) {

        setStatus(
            "LiveKit SDK not loaded"
        );

        console.error(
            "LivekitClient is undefined."
        );

        return;
    }


    /*
     * Check credentials.
     */

    if (
        !LIVEKIT_URL ||
        !TOKEN
    ) {

        setStatus(
            "Missing room credentials"
        );

        console.error(
            "LIVEKIT_URL or TOKEN is missing."
        );

        return;
    }


    try {

        setStatus(
            "Connecting..."
        );


        /* ----------------------------------------------------
           Create LiveKit room
           ---------------------------------------------------- */

        room =
            new LivekitClient.Room({

                /*
                 * Automatically manage
                 * video quality based
                 * on available bandwidth.
                 */

                adaptiveStream: true,

                /*
                 * Only send video layers
                 * that are actually needed.
                 */

                dynacast: true
            });


        /* ----------------------------------------------------
           Room connected
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent.Connected,
            () => {

                console.log(
                    "Connected to LiveKit room."
                );

                setStatus(
                    "LIVE",
                    true
                );
            }
        );


        /* ----------------------------------------------------
           Room disconnected
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent.Disconnected,
            reason => {

                console.log(
                    "Disconnected:",
                    reason
                );

                setStatus(
                    "Call ended"
                );

                microphoneEnabled =
                    false;

                if (muteButton) {
                    muteButton.disabled =
                        true;
                }

                if (leaveButton) {
                    leaveButton.disabled =
                        true;
                }
            }
        );


        /* ----------------------------------------------------
           Participant joined
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent
                .ParticipantConnected,

            participant => {

                console.log(
                    "Participant joined:",
                    participant.identity
                );

                setupExistingParticipant(
                    participant
                );
            }
        );


        /* ----------------------------------------------------
           Participant left
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent
                .ParticipantDisconnected,

            participant => {

                console.log(
                    "Participant left:",
                    participant.identity
                );

                removeParticipantCard(
                    participant
                );
            }
        );


        /* ----------------------------------------------------
           Track subscribed
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent
                .TrackSubscribed,

            (
                track,
                publication,
                participant
            ) => {

                console.log(
                    "Track subscribed:",
                    track.kind,
                    participant.identity
                );

                attachTrack(
                    track,
                    participant
                );
            }
        );


        /* ----------------------------------------------------
           Track unsubscribed
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent
                .TrackUnsubscribed,

            (
                track,
                publication,
                participant
            ) => {

                console.log(
                    "Track unsubscribed:",
                    track.kind,
                    participant.identity
                );

                detachTrack(track);
            }
        );


        /* ----------------------------------------------------
           Speaking detection
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent
                .ActiveSpeakersChanged,

            speakers => {

                updateSpeakingIndicators(
                    speakers
                );
            }
        );


        /* ----------------------------------------------------
           Connection state
           ---------------------------------------------------- */

        room.on(
            LivekitClient.RoomEvent
                .ConnectionStateChanged,

            state => {

                console.log(
                    "Connection state:",
                    state
                );

                if (
                    state ===
                    LivekitClient.ConnectionState
                        .Reconnecting
                ) {

                    setStatus(
                        "Reconnecting..."
                    );

                } else if (
                    state ===
                    LivekitClient.ConnectionState
                        .Connected
                ) {

                    setStatus(
                        "LIVE",
                        true
                    );
                }
            }
        );


        /* ----------------------------------------------------
           Connect
           ---------------------------------------------------- */

        await room.connect(
            LIVEKIT_URL,
            TOKEN
        );


        console.log(
            "Room connected:",
            room.name
        );


        /* ----------------------------------------------------
           Enable microphone
           ---------------------------------------------------- */

        await room.localParticipant
            .setMicrophoneEnabled(
                true
            );

        microphoneEnabled = true;


        /* ----------------------------------------------------
           Add participants already
           in the room
           ---------------------------------------------------- */

        room.remoteParticipants
            .forEach(
                participant => {

                    setupExistingParticipant(
                        participant
                    );
                }
            );


    } catch (error) {

        console.error(
            "LiveKit connection failed:",
            error
        );

        setStatus(
            "Connection failed"
        );
    }
}


/* ============================================================
   MUTE / UNMUTE
   ============================================================ */

if (muteButton) {

    muteButton.addEventListener(
        "click",
        async () => {

            if (!room) {
                return;
            }


            try {

                microphoneEnabled =
                    !microphoneEnabled;


                await room.localParticipant
                    .setMicrophoneEnabled(
                        microphoneEnabled
                    );


                muteButton.textContent =
                    microphoneEnabled
                        ? "🎤 Mute"
                        : "🔇 Unmute";


            } catch (error) {

                console.error(
                    "Microphone error:",
                    error
                );
            }
        }
    );
}


/* ============================================================
   END CALL
   ============================================================ */

if (leaveButton) {

    leaveButton.addEventListener(
        "click",
        async () => {

            if (!room) {
                return;
            }


            try {

                await room.disconnect();

                setStatus(
                    "Call ended"
                );


                muteButton.disabled =
                    true;

                leaveButton.disabled =
                    true;


            } catch (error) {

                console.error(
                    "Failed to end call:",
                    error
                );
            }
        }
    );
}


/* ============================================================
   CLEANUP WHEN PAGE IS CLOSED
   ============================================================ */

window.addEventListener(
    "beforeunload",
    () => {

        if (room) {
            room.disconnect();
        }
    }
);


/* ============================================================
   START
   ============================================================ */

connectToRoom();