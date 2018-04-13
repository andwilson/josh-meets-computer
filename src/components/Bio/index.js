import React from 'react';
import styled from 'styled-components';

import avatar from '../../images/avatar.jpg'

const Img = styled.img`
  height: 100px;
`;

class Bio extends React.Component {
  render() {
    return (
      <Img
        src={avatar}
        alt={`Josh Zastrow`}
      />
    )
  }
}

export default Bio
