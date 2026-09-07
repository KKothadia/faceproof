// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract EvidenceRegistry {
    struct Record {
        bytes32 evidenceHash;
        bytes32 mediaHash;
        uint64 timestamp;
        address submitter;
    }
    
    mapping(bytes32 => Record) public records;
    
    event EvidenceAnchored(
        bytes32 indexed evidenceHash,
        bytes32 indexed mediaHash,
        uint64 timestamp,
        address indexed submitter
    );
    
    function anchorEvidence(
        bytes32 evidenceHash,
        bytes32 mediaHash
    ) external {
        require(evidenceHash != bytes32(0), "empty evidence hash");
        require(records[evidenceHash].timestamp == 0, "already anchored");
        
        records[evidenceHash] = Record({
            evidenceHash: evidenceHash,
            mediaHash: mediaHash,
            timestamp: uint64(block.timestamp),
            submitter: msg.sender
        });
        
        emit EvidenceAnchored(
            evidenceHash,
            mediaHash,
            uint64(block.timestamp),
            msg.sender
        );
    }
    
    function verifyEvidence(bytes32 evidenceHash) external view returns (Record memory) {
        require(records[evidenceHash].timestamp != 0, "not found");
        return records[evidenceHash];
    }
}
